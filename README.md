# 📄 AI-Powered PDF Chatbot (RAG)

## 🚀 Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDFs and ask questions. The system retrieves relevant document chunks and generates accurate answers using Groq LLM.

## 🧠 Tech Stack

* Python
* Streamlit
* FAISS (Vector Search)
* Groq API (LLM)
* PyPDF2

## ⚙️ Features

* Upload any PDF
* Ask questions in natural language
* Context-aware AI responses
* ChatGPT-like UI
* Fast semantic search using embeddings

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔐 Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

## 💡 Use Case

* Resume analysis
* Research papers Q&A
* Notes summarization
