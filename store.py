import faiss
import numpy as np

def store_in_faiss(embeddings,embedding_dim):
    embeddings=np.array(embeddings).astype('float')
    index=faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def retrieve_from_faiss(index,query_embedding, k=3):
    query_embedding=np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)  
    return distances, indices