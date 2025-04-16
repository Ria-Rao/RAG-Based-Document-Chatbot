# RAG-Based-Document-Chatbot

This project is a lightweight Retrieval-Augmented Generation (RAG) chatbot that allows users to query a PDF document using natural language. It leverages FAISS for semantic similarity search, Sentence Transformers for generating text embeddings, and Streamlit for an interactive frontend interface.

---

## Features

- Upload and process a PDF document
- Extract and chunk the text for embedding
- Generate embeddings using Sentence Transformers
- Store embeddings in a FAISS index for efficient retrieval
- Query the document and receive top-matching responses

---

## Tech Stack

- Python
- Streamlit
- PyMuPDF (fitz)
- FAISS
- Sentence Transformers
