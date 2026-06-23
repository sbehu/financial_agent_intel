import os
import logging
from openai import OpenAI
import chromadb
import pdfplumber
from dotenv import load_dotenv

# Silence third-party telemetry logs
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# 1. Establish absolute project root pathway
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Extract configuration tokens
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# 3. Instantiate structural connections
openai_client = OpenAI()
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_storage")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("financial_analysis_pool")

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Target PDF file not found at: {pdf_path}")
        
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text(layout=True)
            if page_text and page_text.strip():
                text_chunks.append((page_text.strip(), page_num + 1))
    return text_chunks

def ingest_explicit_pdf(full_target_path, company, year, batch_size=32):
    filename = os.path.basename(full_target_path)
    
    print(f"\n📄 Processing Document: {filename} ({company} - {year})...")
    pages_data = extract_text_from_pdf(full_target_path)
    
    raw_paragraphs = []
    raw_metadatas = []
    
    # Step 1: Flatten out all paragraphs and prepare metadata list
    for text, page_number in pages_data:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paragraphs:
            if len(p) < 40:  # Clean out noise artifacts
                continue
                
            # 🔥 SAFETY GUARDRAIL: If a chunk is exceptionally long, split it up safely
            # 8000 characters is a safe limit well below the 8192 token max.
            if len(p) > 8000:
                sub_chunks = [p[i : i + 8000] for i in range(0, len(p), 8000)]
            else:
                sub_chunks = [p]
                
            for chunk in sub_chunks:
                raw_paragraphs.append(chunk)
                raw_metadatas.append({
                    "source": filename,
                    "filename": filename,
                    "page": page_number,
                    "company": str(company).strip(),
                    "year": int(year)
                })
            
    total_chunks = len(raw_paragraphs)
    print(f"📦 Chunks found: {total_chunks}. Generating vector embeddings...")

    if total_chunks == 0:
        print("⚠️ No content extracted.")
        return

    for i in range(0, total_chunks, batch_size):
        batch_text = raw_paragraphs[i : i + batch_size]
        batch_meta = raw_metadatas[i : i + batch_size]
        batch_ids = [f"{filename}_chunk_{j}" for j in range(i, min(i + batch_size, total_chunks))]
        
        response = openai_client.embeddings.create(
            input=batch_text,
            model="text-embedding-3-small"
        )
        batch_embeddings = [record.embedding for record in response.data]
        
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_text,
            metadatas=batch_meta
        )
        
    print(f"✅ Successfully loaded {filename} into vector pool!")

if __name__ == "__main__":
    print("🚀 Initializing mass data ingestion from your main data folder...")
    
    # Strictly maps to your main 'data' folder files
    files_to_load = [
        {"path": os.path.join(BASE_DIR, "data", "Amex_2020.pdf"), "company": "Amex", "year": 2020},
        {"path": os.path.join(BASE_DIR, "data", "Amex_2021.pdf"), "company": "Amex", "year": 2021},
        {"path": os.path.join(BASE_DIR, "data", "Amex_2022.pdf"), "company": "Amex", "year": 2022},
        {"path": os.path.join(BASE_DIR, "data", "Amex_2023.pdf"), "company": "Amex", "year": 2023},
        {"path": os.path.join(BASE_DIR, "data", "Amex_2024.pdf"), "company": "Amex", "year": 2024},
        {"path": os.path.join(BASE_DIR, "data", "Amex_2025.pdf"), "company": "Amex", "year": 2025},
    ]
    
    # One loop triggers them all sequentially in a single execution run
    for doc in files_to_load:
        try:
            ingest_explicit_pdf(doc["path"], doc["company"], doc["year"])
        except Exception as e:
            print(f"❌ Ingestion broke at {os.path.basename(doc['path'])}: {str(e)}")
            
    print("\n🎉 ALL DATA HAS BEEN COMPLETELY CONSUMED AND RUN!")