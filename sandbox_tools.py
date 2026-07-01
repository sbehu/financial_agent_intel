import os
import json
import re
import requests
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from tavily import TavilyClient
# 🌟 BYPASS BROKEN CREWAI IMPORTS NATIVELY VIA LANGCHAIN CORE
from langchain_core.tools import tool

# Load environment configurations
load_dotenv()
client = OpenAI()

# Initialize connection to our persistent local vector vault
DB_PATH = "chroma_db_storage"


def get_internal_vault_collection():
    """Safely initializes and returns the vector collection only when called."""
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        return chroma_client.get_collection(name="amex_historical_index")
    except Exception:
        # Fallback to get_or_create to prevent crashing if uninitialized in cloud
        try:
            chroma_client = chromadb.PersistentClient(path=DB_PATH)
            return chroma_client.get_or_create_collection(name="amex_historical_index")
        except Exception as e:
            print(f"⚠️ Vector database critical storage lock: {e}")
            return None


# ──── TOOL 1: INTERNAL VECTOR VAULT SEARCH ENGINE ────

@tool("query_internal_amex_vault")
def query_internal_amex_vault(search_query: str) -> str:
    """Queries the local vector database chunk array for American Express corporate records using type-routing."""
    try:
        print("\n==================================================")
        print("🔌 RETRIEVAL TOOL ACTIVE LOG TRIGGERED")
        print(f"👉 Raw Argument Value Received: '{search_query}'")
        print("==================================================")
        collection = get_internal_vault_collection()

        if collection is None:
            return "⚠️ Internal Vault database is temporarily unreachable in this cluster environment."
        

        if not search_query or not str(search_query).strip():
            return "No search query provided to the retrieval system."

        table_indicators = ["interest", "income", "revenue", "growth", "scaling", "trend", "yoy", "percentage", "amount", "balance"]
        is_tabular_request = any(indicator in search_query.lower() for indicator in table_indicators)
        
        if is_tabular_request:
            optimized_search = f"{search_query} [START_STRUCTURAL_TABLE]"
        else:
            stop_phrases = ["what are the", "what is the", "as per the", "according to", "of the", "tell me about"]
            cleaned_query = search_query.lower()
            for phrase in stop_phrases:
                cleaned_query = cleaned_query.replace(phrase, "")
            optimized_search = " ".join(cleaned_query.split()).strip()
            
        results = collection.query(query_texts=[optimized_search], n_results=5)
        
        compiled_output = []
        for i, document in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] else {}
            page_num = metadata.get('page', 'UNKNOWN_PAGE')
            source_file = metadata.get('source', 'UNKNOWN_FILE')
            
            chunk_block = (
                f"--- INTERNAL SOURCE: {source_file} (PAGE {page_num}) ---\n"
                f"{document}\n"
                f"--------------------------------------------------"
            )
            compiled_output.append(chunk_block)
            
        return "\n\n".join(compiled_output) if compiled_output else "No relevant records found."
    except Exception as e:
        return f"⚠️ Internal Vault Lookup Error: {e}"


# ──── TOOL 2: LIVE WEB COMPETITOR BENCHMARKER ────

@tool("query_external_competitor_web")
def query_external_competitor_web(competitor_query: str) -> str:
    """Queries external market data channels using a live web search pipeline to look up up-to-date competitive interest rates or bank metrics."""
    try:
        print(f"🌐 Activating Live Web Search Pipeline for: '{competitor_query}'...")
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily.search(
            query=f"{competitor_query} interest income trend reports",
            search_depth="advanced",
            max_results=3
        )
        web_context = []
        for result in response.get('results', []):
            web_context.append(
                f"--- LIVE WEB SOURCE: {result['title']} ({result['url']}) ---\n"
                f"EXTRACTED VALUE CONTEXT:\n{result['content']}\n"
                f"--------------------------------------------------"
            )
        return "\n\n".join(web_context) if web_context else "No active real-time records found on the web."
    except Exception as e:
        return f"⚠️ Live Web Search API Error: {e}"


# ──── TOOL 3: DETERMINISTIC FINANCIAL ARITHMETIC CORE ────

@tool("execute_financial_calculation")
def execute_financial_calculation(formula_type: str, arguments_json: str) -> str:
    """Deterministic Math Engine to calculate financial ratios without LLM hallucination. formula_type must be 'percentage_change' or 'cagr'."""
    try:
        args = json.loads(arguments_json)
        if formula_type == "percentage_change":
            old_val, new_val = float(args[0]), float(args[1])
            if old_val == 0: return "⚠️ Mathematical Error: Division by zero."
            delta = ((new_val - old_val) / old_val) * 100
            return f"--- DETERMINISTIC MATH ENGINE RESULT ---\nPercentage Change Calculated: {delta:.2f}%\n"
        elif formula_type == "cagr":
            start_val, end_val, years = float(args[0]), float(args[1]), float(args[2])
            if start_val <= 0 or end_val <= 0 or years <= 0: return "⚠️ Mathematical Error: Invalid parameters."
            cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
            return f"--- DETERMINISTIC MATH ENGINE RESULT ---\nCompound Annual Growth Rate (CAGR) Calculated: {cagr:.2f}%\n"
        return f"⚠️ Calculation Error: Formula type '{formula_type}' is not supported."
    except Exception as e:
        return f"⚠️ Calculation Engine Runtime Exception: {e}"


# ──── TOOL 4: REAL-TIME GLOBAL MARKET EXCHANGE RATES ────

@tool("get_live_exchange_rate")
def get_live_exchange_rate(base_currency: str = "USD", target_currency: str = "INR") -> str:    
    """Queries a live market data API to get the real-time exchange rate between two currencies to normalize financial statements."""
    try:
        base = base_currency.upper().strip()
        target = target_currency.upper().strip()
        url = f"https://open.er-api.com/v6/latest/{base}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            rates = response.json().get("rates", {})
            if target in rates:
                return f"--- LIVE EXCHANGE RATE ENGINE --- \n1 {base} = {rates[target]:.4f} {target}\n"
        return "⚠️ Exchange Rate API temporarily unreachable."
    except Exception as e:
        return f"⚠️ Live Currency API Exception: {e}"


# ──── STANDALONE UNIT TESTS VALIDATION MATRIX ────
if __name__ == "__main__":
    print("\n🔬 STARTING STANDALONE TOOLBOX VALIDATION MATRIX...\n")
    print("📋 [TEST 1] Querying Local ChromaDB Vault (Amex Historicals)...")
    internal_query = "Total Interest Income 2024"
    test_result = query_internal_amex_vault.invoke(internal_query)
    print(test_result[:600])