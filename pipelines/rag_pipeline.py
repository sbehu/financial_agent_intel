import os
import logging

# 🤫 SILENCE THIRD-PARTY TELEMETRY WARNINGS
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

from openai import OpenAI
from dotenv import load_dotenv
from tools.vector_db import query_vector_db

# 1. Coordinate pathways from this script up to the root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Extract configuration tokens from your root secret file
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# 3. Initialize the official OpenAI client handle
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def format_context(raw_results):
    """
    Transforms the tuple (documents, metadatas) returned by query_vector_db
    into a structured, human-and-LLM-readable text block.
    """
    if not raw_results or len(raw_results) < 2:
        return ""
    
    documents = raw_results[0]
    metadatas = raw_results[1]
    
    if not documents:
        return ""
        
    formatted_chunks = []
    
    for idx, (text, meta) in enumerate(zip(documents, metadatas), start=1):
        if not text or not meta:
            continue
        source_file = os.path.basename(meta.get('source', 'Unknown File'))
        page_num = meta.get('page', 'N/A')
        
        chunk_header = f"--- CHUNK {idx} | SOURCE: {source_file} (Page {page_num}) ---"
        formatted_chunk = f"{chunk_header}\n{str(text).strip()}"
        
        formatted_chunks.append(formatted_chunk)
    
    return "\n\n".join(formatted_chunks)


def build_prompt(question, formatted_context):
    """
    Constructs a structured system prompt and user payload that allows for hybrid
    internal-external data analysis while protecting ground-truth data authority.
    """
    system_instruction = (
        "You are an expert financial research analyst specializing in corporate equity and benchmarking.\n"
        "You are working with a hybrid information pipeline containing two distinct data pools:\n"
        "1. LOCAL INTERNAL DATA: The absolute ground-truth for American Express metrics. Treat this as flawless.\n"
        "2. EXTERNAL WEB/REFERENCE DATA: High-level contextual info used to evaluate external peers (e.g., Axis Bank, Visa).\n\n"
        "CRITICAL RULES:\n"
        "1. Never guess or hallucinate metrics. If the context contains a brief high-level overview of an external peer rather than granular tables, "
        "provide a qualitative comparative summary using what is present. Do not default to 'I cannot answer' if basic context exists.\n"
        "2. Always cite your data source. If a fact comes from local files, use the file and page number (e.g., [Amex_2020.pdf, Page 4]). "
        "If it comes from external reference web text, acknowledge it cleanly (e.g., [Wikipedia Reference]).\n"
        "3. If both data pools are completely empty or lack any baseline mention of the entities, state clearly: "
        "'I cannot answer this question based on the retrieved document context.'\n"
        "4. Keep your tone clinical, professional, and structurally organized."
    )
    
    user_payload = (
        f"Retrieved Hybrid Context Documentation:\n"
        f"=========================================\n"
        f"{formatted_context}\n"
        f"=========================================\n\n"
        f"User Financial Inquiry: {question}\n"
    )
    
    return system_instruction, user_payload

def execute_rag(user_query, metadata_filter=None):
    """
    The master orchestrator function with intermediate debug prints.
    Accepts an optional metadata_filter dictionary.
    """
    try:
        # STEP 1: Run semantic search against the database safely
        raw_db_output = query_vector_db(user_query, n_results=3, metadata_filter=metadata_filter)
    except Exception as e:
        print(f"⚠️ [DEBUG] Database read bypassed or index uninitialized: {str(e)}")
        return ""
        
    # STEP 2: Format raw database lists into a clean text block
    context_block = format_context(raw_db_output)
    
    # 🚀 FIX: If context is completely blank, return empty string instantly 
    # to let financial_agent.py switch directly to real-time web search!
    if not context_block.strip():
        return ""
        
    # STEP 3 & 4: Map inputs into system instructions and user payloads
    system_instruction, user_payload = build_prompt(user_query, context_block)
    
    try:
        # STEP 5: Send the structured payload to OpenAI safely
        print("📡 Dispatching network call to OpenAI API...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}  # 🔥 FIXED VALID SYSTEM ROLE LAYOUT
            ],
            temperature=0.0 
        )
        
        ai_response = response.choices[0].message.content
        
        # 🛡️ STEP 7: Run the Groundedness Guardrail Check
        print("🛡️ Guardrail active: Auditing response for hallucinations...")
        audit_result = verify_groundedness(context_block, ai_response)
        
        if "SCORE: FAILED" in audit_result:
            return f"Guardrail Warning: The generated answer failed validation safety checks.\nDetails: {audit_result}"
            
        return ai_response
        
    except Exception as e:
        return f"An operational error occurred during API execution: {str(e)}"
        
def verify_groundedness(context_block, ai_response):
    """
    Acts as a strict financial auditor. Checks if the AI's response
    is 100% supported by the extracted database text chunks.
    """
    audit_instruction = (
        "You are a strict financial auditor.\n"
        "Your sole task is to verify if the 'Draft Answer' is completely grounded in the provided 'Database Context'.\n\n"
        "CRITICAL AUDIT RULES:\n"
        "1. Check every single number, percentage, monetary value, and financial claim in the Draft Answer.\n"
        "2. If any number or claim is NOT explicitly mentioned in the Database Context, or if it contradicts the context, the audit fails.\n"
        "3. Do not use external knowledge. If the context says $1,015 million, and the answer says $1,015 billion or a completely different number, it fails.\n\n"
        "Output exactly one of these two formats:\n"
        "1. If it passes all rules: 'SCORE: PASSED'\n"
        "2. If it fails any rule: 'SCORE: FAILED | Reason: [Briefly explain what number or fact was hallucinated]'"
    )
    
    audit_payload = (
        f"Database Context:\n"
        f"-----------------\n"
        f"{context_block}\n"
        f"-----------------\n\n"
        f"Draft Answer:\n"
        f"-------------\n"
        f"{ai_response}\n"
        f"-------------\n"
    )
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": audit_instruction},
                {"role": "user", "content": audit_payload}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"SCORE: ERROR | Guardrail API issue: {str(e)}"

# ==========================================
# 🧪 LIVE PIPELINE FILTER VERIFICATION
# ==========================================
if __name__ == "__main__":
    print("🚀 Running pipeline with metadata filtering enabled...")
    test_query = "What was the total provision for losses for the year ended December 31, 2020?"
    final_answer = execute_rag(test_query, metadata_filter={"year": 2020})
    print("==================== FINAL OUTPUT RETURNED TO USER ====================")
    print(final_answer)
    print("=======================================================================")