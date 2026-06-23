import os
import chromadb
from dotenv import load_dotenv

# 1. Coordinate paths from this script file up to the parent root workspace
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Extract configuration tokens from your root .env file
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# 3. Instantiate the connection bridge to your storage directory
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_storage")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection_stats(collection_name):
    """
    Looks inside a specific collection drawer and prints how many total 
    paragraphs are currently stored inside it.
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        count = collection.count()
        print(f"📊 Collection Name: '{collection_name}' | Total Active Chunks: {count}")
        return count
    except ValueError:
        print(f"⚠️ Collection '{collection_name}' does not exist on disk.")
        return None

def clear_collection(collection_name):
    """
    Safely disconnects and deletes a specific collection drawer from disk 
    without corrupting the main database map files.
    """
    try:
        chroma_client.delete_collection(collection_name)
        print(f"🗑️ Successfully deleted collection '{collection_name}' from storage.")
        return True
    except ValueError:
        print(f"⚠️ Cannot delete: Collection '{collection_name}' was not found.")
        return False


# ==========================================
# 🧪 LOCAL TESTING BLOCK (Temporary)
# ==========================================
# if __name__ == "__main__":
#     print("🚀 Initializing standalone validation for db_manage.py...")
    
#     print("\n--- Step 1: Checking your real database stats ---")
#     get_collection_stats("amex_financial_data")
    
#     print("\n--- Step 2: Checking the accidental empty placeholder collection ---")
#     get_collection_stats("financial_reports")
    
#     print("\n--- Step 3: Cleaning up the empty placeholder drawer ---")
#     # This safely deletes the accidental placeholder and its folder (47eda087...)
#     clear_collection("financial_reports")
    
#     print("\n--- Step 4: Verifying final remaining collections on disk ---")
#     try:
#         existing_collections = chroma_client.list_collections()
#         print(f"📋 Final remaining collections on disk: {[c.name for c in existing_collections]}")
#     except Exception as e:
#         print(f"❌ Error listing final collections: {e}")