from src.document_processing import parse_document, clean_text, chunk_text

def main():
    file_path = "testfiles/powerpoints/samplepptx.pptx"
    
    parsed_text = parse_document(file_path)
    
    if parsed_text:
        cleaned_text = clean_text(parsed_text)
        chunks = chunk_text(cleaned_text)
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk}\n")
    else:
        print("Failed to parse the document.")

if __name__ == "__main__":
    main()
