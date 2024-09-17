import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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
    
    index.add(embeddings)
    return index

def search_query(index, query_embedding, top_k=5):
    """
    Searches for the most similar text chunks to the query in the Faiss index.
    Returns 2D arrays for both indices and distances.
    """
    distances, indices = index.search(query_embedding, top_k)
    return indices.reshape(1, -1), distances.reshape(1, -1)

def improved_search_query(index, query_embedding, chunks, query, top_k=5):
    """
    Performs an improved search considering both vector similarity and TF-IDF.
    """

    distances, indices = index.search(query_embedding, top_k * 2)  # Get more initial results

    tfidf = TfidfVectorizer().fit_transform(chunks + [query])
    chunk_vectors = tfidf[:-1]
    query_vector = tfidf[-1]
    
    tfidf_similarities = chunk_vectors.dot(query_vector.T).toarray().flatten()

    combined_scores = []
    for i, idx in enumerate(indices[0]):
        vector_sim = 1 / (1 + distances[0][i])
        tfidf_sim = tfidf_similarities[idx]
        combined_score = (vector_sim + tfidf_sim) / 2
        combined_scores.append((idx, combined_score))
    
    top_chunks = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    return [idx for idx, _ in top_chunks], [score for _, score in top_chunks]