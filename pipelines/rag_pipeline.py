import os



import logging
# 🤫 SILENCE THIRD-PARTY TELEMETRY WARNINGS
# We force the chromadb logger to only show critical errors, hiding the telemetry warnings.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

from openai import OpenAI
from dotenv import load_dotenv
# Import the database retrieval capability we built in Stage 1
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
        source_file = os.path.basename(meta.get('source', 'Unknown File'))
        page_num = meta.get('page', 'N/A')
        
        chunk_header = f"--- CHUNK {idx} | SOURCE: {source_file} (Page {page_num}) ---"
        formatted_chunk = f"{chunk_header}\n{text.strip()}"
        
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
def execute_rag(user_query,metadata_filter=None):
    """
    The master orchestrator function with intermediate debug prints.
    Accepts an optional metadata_filter dictionary.
    """
    # STEP 1: Run semantic search against the database
    raw_db_output = query_vector_db(user_query, n_results=3,metadata_filter=metadata_filter)
    
    # 🔍 NEW CHECKPOINT: See the raw, untouched data from ChromaDB
    print("\n========================= [DEBUG] STEP 1: RAW_DB_OUTPUT VALUE =========================")
    import pprint
    pprint.pprint(raw_db_output)
    print("======================================================================================\n")
    
    # STEP 2: Format raw database lists into a clean text block
    context_block = format_context(raw_db_output)
    
    # 🔍 CHECKPOINT 1: See what the text looks like after format_context finishes
    print("\n========================= [DEBUG] STEP 2: CONTEXT_BLOCK VALUE =========================")
    print(context_block)
    print("======================================================================================\n")
    
    if not context_block:
        return "I cannot find any relevant background documentation in the system to answer this question."
        
    # STEP 3 & 4: Map inputs into system instructions and user payloads
    system_instruction, user_payload = build_prompt(user_query, context_block)
    
    # 🔍 CHECKPOINT 2: See exactly what we are handing over to the User Payload
    print("========================= [DEBUG] STEP 4: USER_PAYLOAD VALUE =========================")
    print(user_payload)
    print("======================================================================================\n")
    
    try:
        # STEP 5: Send the structured payload to OpenAI
        print("📡 Dispatching network call to OpenAI API...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ],
            temperature=0.0 
        )
        
        # 🔍 CHECKPOINT 3: See the raw data package OpenAI sends back before we parse it
        print("\n========================= [DEBUG] STEP 5: RAW RESPONSE FROM OPENAI =========================")
        print(response)
        print("============================================================================================\n")
        
        # Step 6: Extract and return the clean text answer string
        # Step 6: Extract the clean text answer string
        ai_response = response.choices[0].message.content
        
        # 🛡️ STEP 7: Run the Groundedness Guardrail Check
        print("🛡️ Guardrail active: Auditing response for hallucinations...")
        audit_result = verify_groundedness(context_block, ai_response)
        
        print("\n========================= [DEBUG] STEP 7: GUARDRAIL SCORE =========================")
        print(audit_result)
        print("==================================================================================\n")
        
        # If the audit fails, we stop the execution and return a safe message instead of the hallucination
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
        "- If it passes all rules: 'SCORE: PASSED'\n"
        "- If it fails any rule: 'SCORE: FAILED | Reason: [Briefly explain what number or fact was hallucinated]'"
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
# 🧪 LIVE PIPELINE VERIFICATION
# ==========================================
# if __name__ == "__main__":
#     print("🚀 Running pipeline with intermediate debugging enabled...")
#     test_query = "What was the total provision for losses for the year ended December 31, 2020?"
    
#     final_answer = execute_rag(test_query)
    
#     print("==================== FINAL OUTPUT RETURNED TO USER ====================")
#     print(final_answer)
#     print("=======================================================================")


# ==========================================
# 🧪 DELIBERATE HALLUCINATION TEST
# ==========================================
# if __name__ == "__main__":
#     print("🚀 Testing Guardrail with a deliberate fake number...")
    
#     # Fake context block containing true data
#     fake_context = "Provisions for Card Member receivables: $1,015 million."
    
#     # Fake answer containing a complete hallucination
#     fake_answer = "The provision for Card Member receivables was $9,999 million."
    
#     print("🛡️ Sending to auditor...")
#     audit_result = verify_groundedness(fake_context, fake_answer)
    
#     print("\n========================= GUARDRAIL OUTPUT =========================")
#     print(audit_result)
#     print("====================================================================\n")    


# ==========================================
# 🧪 LIVE PIPELINE FILTER VERIFICATION
# ==========================================
if __name__ == "__main__":
    print("🚀 Running pipeline with metadata filtering enabled...")
    test_query = "What was the total provision for losses for the year ended December 31, 2020?"
    
    # We pass the query AND specify that we only want chunks from 2020
    final_answer = execute_rag(test_query, metadata_filter={"year": 2020})
    
    print("==================== FINAL OUTPUT RETURNED TO USER ====================")
    print(final_answer)
    print("=======================================================================")    