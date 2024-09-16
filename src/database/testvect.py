from src.database.vector_db import create_faiss_index, store_embeddings, search_query
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Ensure this is consistent with your main code

# Example list of document chunks
text_chunks = [
    "Faiss is an efficient similarity search library.",
    "Sentence transformers are useful for creating text embeddings.",
    "Vector search is powerful for finding relevant information.",
    "HELOOO",
    "aisdiasdiasdiaisd"
]

def test_faiss():
    # 1. Create Faiss index
    dimension = 384  # Ensure this matches the embedding size
    index = create_faiss_index(dimension)

    # 2. Store embeddings into the index
    store_embeddings(index, text_chunks, model)  # Pass chunks and model

    # 3. Perform a search query
    query = "What is Faiss?"
    query_embedding = model.encode([query])  # Generate embedding for the query
    indices, distances = search_query(index, query_embedding, top_k=3)

    # Output the results
    print(f"Top {len(indices)} matching chunks:")
    for i in range(len(indices[0])):
        print(f"Chunk {indices[0][i]} with distance {distances[0][i]}: {text_chunks[indices[0][i]]}")

if __name__ == "__main__":
    test_faiss()
