from flask import Flask, render_template, request, jsonify
from src.document_processing import parse_document, clean_text, chunk_text
from src.database.vector_db import create_faiss_index, improved_search_query
from src.embedding.embedding_generator import EmbeddingGenerator, process_and_store_document
from main import extract_key_sentences
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Global variables to store the processed document and FAISS index
processed_chunks = None
faiss_index = None
embedding_generator = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global processed_chunks, faiss_index, embedding_generator
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)
        
        # Process the uploaded file
        try:
            parsed_text = parse_document(temp_path)
            
            if parsed_text:
                cleaned_text = clean_text(parsed_text)
                processed_chunks = chunk_text(cleaned_text, chunk_size=60, overlap=10)
                
                # Initialize EmbeddingGenerator
                embedding_generator = EmbeddingGenerator()
                
                # Create FAISS index
                embedding_dim = embedding_generator.model.get_sentence_embedding_dimension()
                faiss_index = create_faiss_index(embedding_dim)
                
                # Generate and store embeddings
                process_and_store_document(processed_chunks, faiss_index, embedding_generator)
                
                os.remove(temp_path)  # Remove the temporary file
                return jsonify({'message': 'File processed successfully'})
            else:
                os.remove(temp_path)  # Remove the temporary file
                return jsonify({'error': 'Failed to parse the document'})
        except Exception as e:
            os.remove(temp_path)  # Remove the temporary file
            return jsonify({'error': f'Error processing file: {str(e)}'})

@app.route('/query', methods=['POST'])
def process_query():
    global processed_chunks, faiss_index, embedding_generator
    
    if not processed_chunks or not faiss_index or not embedding_generator:
        return jsonify({'error': 'Please upload a document first'})
    
    user_query = request.json['query']
    query_embedding = embedding_generator.generate_embeddings([user_query])[0]
    query_embedding = query_embedding.reshape(1, -1)
    
    top_chunk_indices, relevance_scores = improved_search_query(faiss_index, query_embedding, processed_chunks, user_query, top_k=3)
    
    relevant_chunks = [processed_chunks[idx] for idx in top_chunk_indices]
    combined_chunks = " ".join(relevant_chunks)
    key_sentences = extract_key_sentences(combined_chunks, num_sentences=3, max_words_per_sentence=40)
    
    response = {
        'relevant_chunks': relevant_chunks,
        'key_sentences': key_sentences
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)