import os
from contextlib import redirect_stderr
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))

ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

openai_client = OpenAI()
CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, "chroma_storage"))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection("financial_analysis_pool")

def query_vector_db(search_query, n_results=8, metadata_filter=None):
    """
    Converts search query to a vector and returns the top N matching text paragraphs safely.
    """
    try:
        response = openai_client.embeddings.create(
            input=[search_query],
            model="text-embedding-3-small"
        )
        query_vector = response.data[0].embedding
        
        with open(os.devnull, 'w') as fnull:
            with redirect_stderr(fnull):
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=n_results,
                    where=metadata_filter
                )
                
        if not results or 'documents' not in results or not results['documents']:
            return [], []
            
        docs = results.get('documents', [[]])[0]
        metas = results.get('metadatas', [[]])[0]
        
        return (docs if docs else []), (metas if metas else [])
        
    except Exception as e:
        print(f"⚠️ Vector DB Operational Query Failure: {str(e)}")
        return [], []