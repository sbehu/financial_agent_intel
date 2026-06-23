import os
from contextlib import redirect_stderr
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

# 1. Establish absolute path tracking from this file to the root directory
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))

# 2. Load env variables from your root .env file
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# 3. Initialize connection clients
openai_client = OpenAI()
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_storage")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# 🎯 TARGETING THE BRAND NEW MULTI-DOC POOL COLLECTION
collection = chroma_client.get_or_create_collection("financial_analysis_pool")

def query_vector_db(search_query, n_results=8, metadata_filter=None):
    """
    Converts search query to a vector and returns the top N matching text paragraphs.
    Accepts an optional metadata_filter dictionary (e.g., {"year": 2020}) for ChromaDB's where clause.
    """
    response = openai_client.embeddings.create(
        input=[search_query],
        model="text-embedding-3-small"
    )
    query_vector = response.data[0].embedding
    
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull):
            # 🚀 METADATA UPDATE: Pass the metadata_filter dict directly to ChromaDB's where parameter
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=metadata_filter  # Defaults to None if no filter is applied
            )
            
    return results.get('documents', [[]])[0], results.get('metadatas', [[]])[0]


# # ==========================================
# # 🧪 LOCAL TESTING BLOCK (Temporary)
# # ==========================================
# if __name__ == "__main__":
#     print("🚀 Fetching exactly 1 sample chunk from your real database...")
    
#     try:
#         # Testing the filter directly
#         documents, metadatas = query_vector_db(
#             "What was the provision for losses?", 
#             n_results=1, 
#             metadata_filter={"year": 2020}
#         )
        
#         if documents and metadatas:
#             print("\n📝 --- EXACT TEXT CHUNK FROM DATABASE ---")
#             print(documents[0])
            
#             print("\n🏷️ --- EXACT METADATA DICTIONARY FROM DATABASE ---")
#             print(metadatas[0])
#         else:
#             print("\n❌ Connected successfully, but no text chunks were returned.")
            
#     except Exception as e:
#         print(f"\n❌ Error encountered during test: {e}")