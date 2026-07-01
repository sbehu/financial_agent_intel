import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Load environment configurations
load_dotenv()

# Initialize standalone OpenAI client
client = OpenAI()

def initialize_vector_store():
    """
    Initializes a persistent local database directory and sets up
    a layout-aware vector collection using OpenAI embeddings.
    """
    db_path = "chroma_db_storage"
    print(f"📦 Initializing Persistent ChromaDB Store at: ./{db_path}")
    
    # Create a persistent local database client (saves data directly to your disk)
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Define our mathematical vector embedding engine
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    
    # Create or fetch our financial index collection
    collection = chroma_client.get_or_create_collection(
        name="amex_historical_index",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"} # Enforces high-precision cosine similarity matching
    )
    
    return collection

def populate_database(collection, chunks):
    """
    Batches and inserts layout-aware text chunks and metadata matrices 
    directly into the persistent vector database.
    """
    if not chunks:
        print("⚠️ No data chunks provided for database injection.")
        return
        
    print(f"\n🚀 Committing {len(chunks)} text layers into Vector Database...")
    
    ids = []
    documents = []
    metadatas = []
    
    for idx, chunk in enumerate(chunks):
        # Generate a clean, structured unique identifier for every single row
        ids.append(f"id_{chunk['metadata']['source']}_p{chunk['metadata']['page']}_{idx}")
        documents.append(chunk["content"])
        
        # Keep metadata flat and strictly typed for database optimization
        metadatas.append({
            "source": chunk["metadata"]["source"],
            "page": int(chunk["metadata"]["page"])
        })
        
    # ChromaDB optimization: We push records in clean batches to handle large datasets smoothly
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        print(f"   📥 Vectorizing data vectors {i} to {end_idx}...")
        
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        
    print("✅ Database Population Task Matrix Completed Successfully.")

if __name__ == "__main__":
    # 1. Connect to our layout-aware parser from sandbox_ingest
    from sandbox_ingest import extract_layout_aware_chunks
    
    data_dir = "data"
    all_chunks = []
    
    if os.path.exists(data_dir):
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        
        # Process files to gather all our layout chunks
        for pdf in pdf_files:
            pdf_full_path = os.path.join(data_dir, pdf)
            all_chunks.extend(extract_layout_aware_chunks(pdf_full_path))
            
        # 2. Initialize our local storage index
        db_collection = initialize_vector_store()
        
        # 3. Permanently load the vectors into disk storage
        populate_database(db_collection, all_chunks)
        
        # 4. Verification Check: Print total count inside database
        print(f"\n📊 Total active documents inside Vector Index: {db_collection.count()}")
    else:
        print(f"❌ ERROR: '{data_dir}' folder not found.")