import streamlit as st
from pdf_loader import extract_from_pdf
from embedding import generate_embeddings
from store import store_in_faiss, retrieve_from_faiss
from generator import generate_answer

def main():
    st.title("RAG-based PDF Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        text = extract_from_pdf(uploaded_file)
        text_chunks = text.split("\n")
        embeddings = generate_embeddings(text_chunks)
        embedding_dim = len(embeddings[0])
        index = store_in_faiss(embeddings, embedding_dim)
        query = st.text_input("Ask a question about the document:")
        if query:
            query_embedding = generate_embeddings([query])[0]
            distances, indices = retrieve_from_faiss(index, query_embedding, k=3)
            top_chunks = [text_chunks[idx] for idx in indices[0]]
            context = "\n".join(top_chunks)
            answer = generate_answer(context, query)
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
