from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
import nltk

from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi

# =========================
# NLTK DOWNLOADS
# =========================

nltk.download('punkt')
nltk.download('punkt_tab')

_model = None


# =========================
# LOAD EMBEDDING MODEL
# =========================

def get_model():

    global _model

    if _model is None:

        _model = SentenceTransformer(
            'all-MiniLM-L6-v2'
        )

    return _model


# =========================
# DOCUMENT-AWARE CHUNKING
# =========================

def chunk_text(
    text,
    source,
    chunk_size=1000,
    overlap=200
):

    sentences = sent_tokenize(text)

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        if (
            len(current_chunk)
            + len(sentence)
            < chunk_size
        ):

            current_chunk += " " + sentence

        else:

            if len(current_chunk.strip()) > 80:

                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source
                })

            overlap_text = current_chunk[-overlap:]

            current_chunk = (
                overlap_text + " " + sentence
            )

    if len(current_chunk.strip()) > 80:

        chunks.append({
            "text": current_chunk.strip(),
            "source": source
        })

    # =========================
    # REMOVE DUPLICATES
    # =========================

    unique_chunks = []

    seen = set()

    for chunk in chunks:

        chunk_text_value = chunk["text"]

        if chunk_text_value not in seen:

            unique_chunks.append(chunk)

            seen.add(chunk_text_value)

    return unique_chunks


# =========================
# CREATE EMBEDDINGS
# =========================

def create_embeddings(chunks):

    model = get_model()

    texts = [

        chunk["text"]

        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    return np.array(
        embeddings
    ).astype("float32")


# =========================
# BUILD FAISS INDEX
# =========================

def build_faiss_index(embeddings):

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    return index


# =========================
# BUILD BM25 INDEX
# =========================

def build_bm25(chunks):

    tokenized_chunks = [

        chunk["text"].lower().split()

        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    return bm25


# =========================
# HYBRID SEARCH
# =========================

def search_faiss(
    query,
    index,
    chunks,
    bm25,
    top_k=12
):

    model = get_model()

    # =========================
    # SEMANTIC SEARCH
    # =========================

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    semantic_results = []

    for score, i in zip(
        scores[0],
        indices[0]
    ):

        if i < len(chunks):

            semantic_results.append({
                "text": chunks[i]["text"],
                "source": chunks[i]["source"],
                "index": i,
                "score": float(score)
            })

    # =========================
    # BM25 SEARCH
    # =========================

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:top_k]

    keyword_results = []

    for i in bm25_indices:

        keyword_results.append({
            "text": chunks[i]["text"],
            "source": chunks[i]["source"],
            "index": i,
            "score": float(bm25_scores[i])
        })

    # =========================
    # MERGE RESULTS
    # =========================

    combined = semantic_results + keyword_results

    # =========================
    # REMOVE DUPLICATES
    # =========================

    unique_results = []

    seen = set()

    for item in combined:

        if item["text"] not in seen:

            unique_results.append(item)

            seen.add(item["text"])

    # =========================
    # GROUP RESULTS BY PDF
    # =========================

    grouped_results = {}

    for item in unique_results:

        source = item["source"]

        if source not in grouped_results:

            grouped_results[source] = []

        grouped_results[source].append(item)

    # =========================
    # TAKE BEST CHUNKS
    # FROM EACH PDF
    # =========================

    balanced_results = []

    for source in grouped_results:

        sorted_chunks = sorted(
            grouped_results[source],
            key=lambda x: x["score"],
            reverse=True
        )

        # best 2 chunks from each pdf
        balanced_results.extend(
            sorted_chunks[:2]
        )

    # =========================
    # FINAL SORT
    # =========================

    balanced_results = sorted(
        balanced_results,
        key=lambda x: x["score"],
        reverse=True
    )

    # =========================
    # FINAL RESULTS
    # =========================

    final_results = [

        (
            item["text"],
            item["source"],
            item["index"]
        )

        for item in balanced_results[:6]
    ]

    if not final_results:

        return [
            (
                "No relevant context found.",
                "Unknown",
                -1
            )
        ]

    return final_results


# =========================
# SAVE VECTOR DATABASE
# =========================

def save_index(
    index,
    chunks,
    path="vector_store"
):

    os.makedirs(
        path,
        exist_ok=True
    )

    faiss.write_index(
        index,
        f"{path}/index.faiss"
    )

    with open(
        f"{path}/chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )


# =========================
# LOAD VECTOR DATABASE
# =========================

def load_index(
    path="vector_store"
):

    index = faiss.read_index(
        f"{path}/index.faiss"
    )

    with open(
        f"{path}/chunks.pkl",
        "rb"
    ) as f:

        chunks = pickle.load(f)

    return index, chunks