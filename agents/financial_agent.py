import os
import json
import logging
import re
from openai import OpenAI
from dotenv import load_dotenv

from utils.logger import setup_custom_logger
logger = setup_custom_logger("FinancialAgent")

# 🤫 SILENCE THIRD-PARTY TELEMETRY WARNINGS
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

from tools.vector_db import query_vector_db
from pipelines.rag_pipeline import format_context, build_prompt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

class FinancialAgent:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_history = []
        
        # 📊 SINGLE SOURCE OF TRUTH TICKER MAPPING
        self.ticker_map = {
            "axis": "AXISBANK.NS",
            "icici": "ICICIBANK.NS",
            "hdfc": "HDFCBANK.NS",
            "sbi": "SBIN.NS",
            "visa": "V",
            "mastercard": "MA",
            "amex": "AXP",
            "american express": "AXP"
        }

    def format_chat_history(self):
        formatted_messages = []
        for turn in self.conversation_history:
            if turn['speaker'] == 'user':
                formatted_messages.append({"role": "user", "content": turn['text']})
            elif turn['speaker'] == 'assistant':
                formatted_messages.append({"role": "assistant", "content": turn['text']})
        return formatted_messages

    def execute_web_search(self, bank_keyword):
        """
        Fetches live financial metrics using a robust, header-authenticated yfinance session.
        """
        import yfinance as yf
        import requests
        
        ticker_symbol = self.ticker_map.get(bank_keyword.lower(), bank_keyword.upper())
        logger.info(f"📡 Querying yfinance for official symbol: {ticker_symbol}")
        
        try:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            ticker_obj = yf.Ticker(ticker_symbol, session=session)
            income_statement = None
            
            try:
                income_statement = ticker_obj.get_income_stmt()
            except Exception:
                pass
                
            if income_statement is None or income_statement.empty:
                income_statement = ticker_obj.income_stmt
            if income_statement is None or income_statement.empty:
                income_statement = ticker_obj.financials

            if income_statement is None or income_statement.empty:
                return f"[Live Web Search Error: No active data frame found for {ticker_symbol}]"
                
            return f"\n--- LIVE WEB MARKET DATA FOR {ticker_symbol} ---\n{income_statement.to_string()}\n"
            
        except Exception as e:
            return f"[Live Web Search Exception for {ticker_symbol}: {str(e)}]"

    def process_query(self, user_query):
        """
        Processes multi-asset requests by creating a hybrid contextual layout 
        blending local RAG matrices and real-time scrapers seamlessly.
        """
        logger.info(f"🚀 Processing query: {user_query}")
        
        # Step 1: Gather local vectorized contexts safely
        local_rag_context = ""
        try:
            local_documents, local_metadatas = query_vector_db(user_query, n_results=6)
            
            if local_documents and len(local_documents) > 0 and local_documents[0] is not None:
                local_rag_context = format_context((local_documents, local_metadatas))
            else:
                logger.warning("⚠️ ChromaDB query returned zero matching documents. Falling back to empty local context.")
                local_rag_context = "No relevant internal company PDF document snippets were found for this request."
        except Exception as db_err:
            logger.error(f"❌ Error querying or formatting local Vector DB: {str(db_err)}")
            local_rag_context = "An internal error occurred while fetching company document context."

        # Step 2: Scan for live market keywords dynamically
        web_contexts = []
        for keyword in self.ticker_map.keys():
            if re.search(r'\b' + re.escape(keyword) + r'\b', user_query.lower()):
                logger.info(f"🎯 Router triggered live web scraper tracking channel: {keyword}")
                web_data = self.execute_web_search(keyword)
                web_contexts.append(web_data)
        
        # Step 3: Unify discovered data channels into a single cohesive payload
        combined_web_context = "\n".join(web_contexts) if web_contexts else "No supplementary live market data requested."
        
        master_context = (
            f"=== LOCAL KNOWLEDGE POOL DATA ===\n{local_rag_context}\n\n"
            f"=== REAL-TIME EXTERNAL WEB DATA ===\n{combined_web_context}"
        )
        
        # Step 4: System synthesis via GPT-4o-mini
        system_prompt, user_payload = build_prompt(user_query, master_context)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.format_chat_history())
        messages.append({"role": "user", "content": user_payload})
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0
            )
            answer = response.choices[0].message.content
            
            self.conversation_history.append({"speaker": "user", "text": user_query})
            self.conversation_history.append({"speaker": "assistant", "text": answer})
            
            return answer
        except Exception as e:
            logger.error(f"❌ Synthesis Pipeline Generation Failure: {str(e)}")
            return f"An operational error occurred during data compilation: {str(e)}"