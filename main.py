from src.document_processing import parse_document, clean_text, chunk_text, print_chunk_debug_info
from src.database.vector_db import create_faiss_index, store_embeddings, search_query
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk

nltk.download('punkt')
nltk.download('stopwords')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def extract_key_sentences(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = [word.lower() for sentence in sentences for word in sentence.split() if word.lower() not in stopwords.words('english')]
    freq_dist = FreqDist(words)
    sentence_scores = {sentence: sum(freq_dist[word.lower()] for word in sentence.split() if word.lower() not in stopwords.words('english')) for sentence in sentences}
    return sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

def main():
    file_path = "testfiles/pdf/pdflatex-4-pages.pdf"
    parsed_text = parse_document(file_path)
    
    if parsed_text:
        print("Parsed text length:", len(parsed_text))
        print("First 500 characters of parsed text:")
        print(parsed_text[:500])
        
        cleaned_text = clean_text(parsed_text)
        print("\nCleaned text length:", len(cleaned_text))
        print("First 500 characters of cleaned text:")
        print(cleaned_text[:500])
        
        chunks = chunk_text(cleaned_text, chunk_size=150, overlap=30)

        print_chunk_debug_info(chunks)

        embedding_dim = model.get_sentence_embedding_dimension()
        faiss_index = create_faiss_index(embedding_dim)

        store_embeddings(faiss_index, chunks, model)
        
        user_query = "What are the main points of this document?"
        query_embedding = model.encode([user_query]).astype('float32')
        
        print(f"\nQuery embedding shape: {query_embedding.shape}")
        print(f"Query embedding mean: {np.mean(query_embedding)}, std: {np.std(query_embedding)}")
        
        top_chunk_indices, distances = search_query(faiss_index, query_embedding, top_k=3)
        
        print("\nMost relevant chunks based on the query:")
        for i, (chunk_idx, distance) in enumerate(zip(top_chunk_indices, distances)):
            print(f"\nChunk {i+1} (distance: {distance:.4f}):")
            print(chunks[chunk_idx])
            
            chunk_embedding = model.encode([chunks[chunk_idx]]).astype('float32')
            print(f"Chunk embedding shape: {chunk_embedding.shape}")
            print(f"Chunk embedding mean: {np.mean(chunk_embedding)}, std: {np.std(chunk_embedding)}")
        
        print("\nKey sentences from the most relevant chunks:")
        combined_chunks = " ".join([chunks[idx] for idx in top_chunk_indices])
        key_sentences = extract_key_sentences(combined_chunks)
        for i, sentence in enumerate(key_sentences, 1):
            print(f"{i}. {sentence}")
    
    else:
        print("Failed to parse the document.")

if __name__ == "__main__":
    main()