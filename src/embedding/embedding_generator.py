from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        embeddings = self.model.encode(chunks)
        return embeddings.astype('float32')

    def store_embeddings(self, embeddings: np.ndarray, faiss_index):
        faiss_index.add(embeddings)

def process_and_store_document(chunks: List[str], faiss_index, embedding_generator: EmbeddingGenerator):
    embeddings = embedding_generator.generate_embeddings(chunks)
    embedding_generator.store_embeddings(embeddings, faiss_index)
    return embeddings
