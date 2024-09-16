import faiss
import numpy as np

def create_faiss_index(dimension):
    """
    Creates a Faiss index with the specified dimension.
    """
    index = faiss.IndexFlatL2(dimension)
    return index

def store_embeddings(index, text_chunks, model):
    """
    Generates and stores embeddings for each text chunk into the Faiss index.
    """
    embeddings = model.encode(text_chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings mean: {np.mean(embeddings)}, std: {np.std(embeddings)}")
    
    index.add(embeddings)
    return index

def search_query(index, query_embedding, top_k=5):
    """
    Searches for the most similar text chunks to the query in the Faiss index.
    """
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]