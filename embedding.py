from sentence_transformers import SentenceTransformer
def generate_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    embeddings = model.encode(text)  
    return embeddings