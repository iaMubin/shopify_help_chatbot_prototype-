import os
from src.parser import parse_pdf_with_links
from src.embedder import chunk_documents, get_or_create_vector_store

def run_ingestion_pipeline(pdf_path: str):
    print(f"Starting Data Ingestion Pipeline for: {pdf_path}")

    # 1. Parse PDF and extract metadata
    print("Step 1: Parsing PDF and extracting links...")
    documents = parse_pdf_with_links(pdf_path)
    print(f"Success: Parsed {len(documents)} pages.")

    # 2. Chunk Documents
    print("Step 2: Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Success: Created {len(chunks)} chunks.")

    # 3. Create/Update Vector Store
    print("Step 3: Embedding and saving to ChromaDB...")
    # Using the same persist directory as our web app
    vectorstore = get_or_create_vector_store(chunks=chunks, persist_dir="vector_store_bge")
    print("\n✅ Pipeline completed successfully! Vector store is updated and ready.")

if __name__ == "__main__":
    # Ensure this path matches the actual file name in your data folder
    PDF_FILE = "data/Shopify Help Center _ Creating and processing returns and exchanges.pdf"

    if os.path.exists(PDF_FILE):
        run_ingestion_pipeline(PDF_FILE)
    else:
        print(f"❌ Error: Could not find the PDF file at '{PDF_FILE}'. Please check the data folder.")
