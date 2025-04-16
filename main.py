import streamlit as st
from pdf_loader import extract_from_pdf
from embedding import generate_embeddings
from store import store_in_faiss, retrieve_from_faiss

def main():
    st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
    st.title("📄💬 Ask Questions from Your PDF")

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Reading and embedding PDF..."):
            text = extract_from_pdf(uploaded_file)
            text_chunks = text.split("\n")
            st.write(f"✅ Total chunks created: {len(text_chunks)}")

            embeddings = generate_embeddings(text_chunks)
            dim = len(embeddings[0])
            index = store_in_faiss(embeddings, dim)

        query = st.text_input("Ask a question:")

        if query:
            query_embedding = generate_embeddings([query])[0]
            distances, indices = retrieve_from_faiss(index, query_embedding, k=3)

            st.subheader("🔍 Top matching chunks:")
            top_chunk = text_chunks[indices[0][0]]
            st.markdown(f"**Answer:** {top_chunk}")

if __name__ == "__main__":
    main()
