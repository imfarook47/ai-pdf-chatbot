import streamlit as st
from pdf_utils import extract_text_from_pdf


from rag import (
    chunk_text,
    create_embeddings,
    build_faiss_index,
    build_bm25,
    search_faiss,
    save_index,
    load_index
)



from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

st.set_page_config(
    page_title="AI PDF Chatbot",
    layout="wide"
)

st.title("AI PDF Chatbot")

# =========================
# SIDEBAR PDF UPLOAD
# =========================

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# =========================
# SESSION STATE
# =========================

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "image_paths" not in st.session_state:
    st.session_state.image_paths = []


# =========================
# LOAD SAVED VECTOR DB
# =========================

if (
    not uploaded_files
    and os.path.exists("vector_store/index.faiss")
    and st.session_state.index is None
):

    index, chunks = load_index()

    st.session_state.index = index
    st.session_state.chunks = chunks

    st.session_state.bm25 = build_bm25(chunks)

    st.success("Loaded existing vector database!")


# =========================
# PROCESS PDFs
# =========================

if uploaded_files:

    with st.spinner("Processing PDFs..."):

        all_chunks = []
        all_image_paths = []

        for uploaded_file in uploaded_files:

            # =========================
            # EXTRACT TEXT
            # =========================

            text = extract_text_from_pdf(uploaded_file)

            text = text[:60000]

            # =========================
            # DOCUMENT-AWARE CHUNKS
            # =========================

            pdf_chunks = chunk_text(
                text,
                uploaded_file.name
            )

            all_chunks.extend(pdf_chunks)

            # =========================
            # EXTRACT IMAGES
            # =========================

            uploaded_file.seek(0)

            image_paths = extract_images_from_pdf(
                uploaded_file
            )

            all_image_paths.extend(image_paths)

        # =========================
        # BUILD VECTOR DB
        # =========================

        embeddings = create_embeddings(all_chunks)

        index = build_faiss_index(embeddings)

        bm25 = build_bm25(all_chunks)

        st.session_state.chunks = all_chunks
        st.session_state.index = index
        st.session_state.bm25 = bm25
        st.session_state.image_paths = all_image_paths

        save_index(index, all_chunks)

    st.success("PDFs processed successfully!")


# =========================
# SIDEBAR IMAGES
# =========================

if st.session_state.image_paths:

    st.sidebar.subheader("Extracted Images")

    for img_path in st.session_state.image_paths[:5]:

        st.sidebar.image(
            img_path,
            use_container_width=True
        )


# =========================
# DISPLAY CHAT
# =========================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# =========================
# CHAT INPUT
# =========================

query = st.chat_input(
    "Ask something about your PDFs..."
)

if (
    query
    and st.session_state.chunks is not None
):

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            query_lower = query.lower()

            is_summary = any(
                word in query_lower
                for word in [
                    "summary",
                    "summarize",
                    "overview",
                    "contents",
                    "about pdf",
                    "about pdfs"
                ]
            )

            # =========================
            # CONTEXT SELECTION
            # =========================

            if is_summary:

                context = "\n\n".join([
                    f"DOCUMENT: {chunk['source']}\n{chunk['text']}"
                    for chunk in st.session_state.chunks[:20]
                ])

                results = []

            else:

                results = search_faiss(
                    query,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.bm25
                )

                if (
                    not results
                    or results[0][2] == -1
                ):

                    context = "\n\n".join([
                        f"DOCUMENT: {chunk['source']}\n{chunk['text']}"
                        for chunk in st.session_state.chunks[:8]
                    ])

                else:

                    context = "\n\n".join([
                        f"DOCUMENT: {r[1]}\n{r[0]}"
                        for r in results[:6]
                    ])

            # =========================
            # CHAT MEMORY
            # =========================

            history = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state.chat_history[-6:]
            ])

            # =========================
            # PROMPT
            # =========================

            prompt = f"""
You are a highly accurate AI assistant.

The context may contain MULTIPLE documents.

STRICT RULES:
1. Answer ONLY from the provided context
2. Mention document names when relevant
3. Do NOT mix unrelated documents
4. If partially available → explain clearly
5. If not available → say "Not available in document"
6. Do NOT guess
7. Do NOT repeat sentences

STYLE:
- Clear explanation
- Use bullet points where helpful
- Keep it concise but informative

Conversation History:
{history}

Context:
{context}

Question:
{query}
"""

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise multi-document assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.1-8b-instant"
            )

            answer = response.choices[0].message.content

            st.write(answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# =========================
# CLEAR CHAT
# =========================

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.bm25 = None
    st.session_state.image_paths = []

    st.rerun()


# =========================
# NO PDF
# =========================

if not uploaded_files:

    st.info(
        "Upload PDFs from sidebar to start chatting"
    )