from src.document_processing import parse_document, clean_text, chunk_text, print_chunk_debug_info
from src.database.vector_db import create_faiss_index, improved_search_query
from src.embedding.embedding_generator import EmbeddingGenerator, process_and_store_document
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


def extract_key_sentences(text, num_sentences=3, max_words_per_sentence=40):
    # Clean and prepare the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    sentences = sent_tokenize(text)
    
    # Remove very short sentences (likely fragments)
    sentences = [s for s in sentences if len(s.split()) > 5]
    
    # Calculate word frequencies
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for sentence in sentences for word in sentence.split() if word.lower() not in stop_words]
    freq_dist = FreqDist(words)
    
    # Score sentences
    sentence_scores = {}
    for sentence in sentences:
        score = sum(freq_dist[word.lower()] for word in sentence.split() if word.lower() not in stop_words)
        sentence_scores[sentence] = score / len(sentence.split())  # Normalize by sentence length
    
    # Select top sentences
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
    # Sort sentences by their original order in the text
    top_sentences.sort(key=lambda s: sentences.index(s))
    
    # Truncate sentences if they're too long
    truncated_sentences = []
    for sentence in top_sentences:
        words = sentence.split()
        if len(words) > max_words_per_sentence:
            truncated = ' '.join(words[:max_words_per_sentence]) + '...'
        else:
            truncated = sentence
        truncated_sentences.append(truncated)
    
    return truncated_sentences

def main():
    file_path = "testfiles/pdf/CS3ID Lecture 3.pdf"
    parsed_text = parse_document(file_path)
    
    if parsed_text:
        cleaned_text = clean_text(parsed_text)
        chunks = chunk_text(cleaned_text, chunk_size=60, overlap=10)

        # Print debug info for chunks
        print_chunk_debug_info(chunks)

        # Initialize EmbeddingGenerator
        embedding_generator = EmbeddingGenerator()
        
        # Create FAISS index
        embedding_dim = embedding_generator.model.get_sentence_embedding_dimension()
        faiss_index = create_faiss_index(embedding_dim)
        
        # Generate and store embeddings
        embeddings = process_and_store_document(chunks, faiss_index, embedding_generator)
        
        user_query = "What are the main topics covered in this lecture?"
        print(f"\nUser Query: {user_query}")
        query_embedding = embedding_generator.generate_embeddings([user_query])[0]  # Get the first (and only) embedding
        query_embedding = query_embedding.reshape(1, -1)  # Reshape to 2D array
        
        # Search for relevant chunks using the improved search
        top_chunk_indices, relevance_scores = improved_search_query(faiss_index, query_embedding, chunks, user_query, top_k=3)
        
        print("\nMost relevant chunks based on the query:")
        for i, (chunk_idx, score) in enumerate(zip(top_chunk_indices, relevance_scores), 1):
            print(f"\nChunk {i} (Relevance Score: {score:.4f}):")
            print(chunks[chunk_idx])
            
        print("\nKey sentences from the most relevant chunks:")
        combined_chunks = " ".join([chunks[idx] for idx in top_chunk_indices])  # Remove [0] indexing
        key_sentences = extract_key_sentences(combined_chunks, num_sentences=3, max_words_per_sentence=30)
        for i, sentence in enumerate(key_sentences, 1):
            print(f"{i}. {sentence}")
        
    else:
        print("Failed to parse the document.")

if __name__ == "__main__":
    main()