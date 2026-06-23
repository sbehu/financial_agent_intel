import os
import json
import logging
import re
from openai import OpenAI
from dotenv import load_dotenv

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
        
        # 📊 TICKER MAPPING LAYER: Connects natural language phrases to active stock symbols
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

    def execute_web_search(self, company_keyword):
        """
        🌐 REAL-TIME FINANCIAL DATA TOOL (yfinance Engine with ISO Date Normalizer):
        Pulls live audited income statements and maps fiscal dates to standard calendar years.
        """
        ticker_symbol = self.ticker_map.get(company_keyword.lower())
        if not ticker_symbol:
            return f"Market Data Error: Could not resolve a public stock ticker for '{company_keyword}'."
        
        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(ticker_symbol)
            
            # Use the absolute most resilient income statement endpoints available
            income_statement = ticker_obj.income_stmt
            if income_statement.empty:
                income_statement = ticker_obj.financials
            if income_statement.empty:
                income_statement = ticker_obj.quarterly_income_stmt
                
            if income_statement.empty:
                return f"No live income statement matrix found for ticker symbol {ticker_symbol}."
            
            # 🚀 FIX: Map full ISO datetime index names (e.g., '2024-03-31 00:00:00') into clean calendar years
            available_years = []
            for col in income_statement.columns:
                col_str = str(col)
                # Regex extraction to pull the first 4 consecutive digits (the year) safely
                year_match = re.search(r'\b(202\d)\b', col_str)
                if year_match:
                    available_years.append(year_match.group(1))
                else:
                    available_years.append(col_str.split("-")[0].strip())
            
            financial_snippet = f"--- LIVE MARKET DATA FOR TICKER: {ticker_symbol} ---\n"
            relevant_rows = [
                "Total Revenue", 
                "Net Income", 
                "Net Interest Income", 
                "Interest Income", 
                "Interest Expense",
                "Operating Revenue"
            ]
            
            # Cross-reference the financial indexes matching our target matrices
            for row in income_statement.index:
                if any(target.lower() in str(row).lower() for target in relevant_rows):
                    values = income_statement.loc[row].values
                    val_strings = []
                    for year, val in zip(available_years, values):
                        # Catch empty rows or non-numeric values safely
                        try:
                            val_strings.append(f"{year}: {val:,.0f}")
                        except (ValueError, TypeError):
                            continue
                    if val_strings:
                        financial_snippet += f"{row} -> {', '.join(val_strings)}\n"
                    
            return financial_snippet
            
        except Exception as e:
            return f"Financial API Stream Error for {ticker_symbol}: {str(e)}"
        
    def chat(self, user_message):
        self.conversation_history.append({'speaker': 'user', 'text': user_message})
        lower_message = user_message.lower()
        
        # 🏢 1. DETERMINE LOCAL VS. HYBRID INTENT (ROUTER LAYER)
        needs_local_amex = False
        needs_web_search = False
        target_peer_keyword = ""
        
        if "amex" in lower_message or "american express" in lower_message or "revenue" in lower_message or "provision" in lower_message:
            needs_local_amex = True
            
        peer_keywords = ["axis", "icici", "hdfc", "visa", "mastercard", "sbi"]
        for peer in peer_keywords:
            if peer in lower_message:
                needs_web_search = True
                target_peer_keyword = peer
                break

        # 📁 2. EXECUTE ROUTED SEARCH CHANNELS
        local_context = ""
        web_context = ""
        
        # Pass A: Query local Vector Database with wide window size
        if needs_local_amex:
            try:
                docs, metas = query_vector_db(user_message, n_results=10, metadata_filter={"company": "Amex"})
                if docs:
                    local_context = format_context([docs, metas])
            except Exception:
                pass
                
        # Pass B: Trigger Live Financial API execution for the active peer asset
        if needs_web_search and target_peer_keyword:
            web_context = self.execute_web_search(target_peer_keyword)

        # 🗃️ 3. CONSOLIDATE CONTEXTS SECURELY
        context_block = f"=== AUTHORITATIVE LOCAL INTERNAL DATA ===\n{local_context}\n\n"
        if web_context:
            context_block += f"=== EXTERNAL REFERENCE WEB DATA ===\n{web_context}"
        
        context_block = context_block.strip()
        
        # 📝 4. CONSTRUCT PROMPT & INJECT HYBRID INSTRUCTIONS
        system_instruction, user_payload = build_prompt(user_message, context_block)
        
        hybrid_rules = (
            "\n\nDATA SOURCE AUTHORITY MATRIX:\n"
            "1. Treat 'AUTHORITATIVE LOCAL INTERNAL DATA' as the absolute ground truth for American Express financials.\n"
            "2. Use 'EXTERNAL REFERENCE WEB DATA' exclusively to extract metrics for requested peer companies.\n"
            "   👉 NOTE ON VALUATION UNITS: Local Amex files report in Millions of USD. External tickers from Yahoo Finance (like .NS symbols) "
            "   report raw financial table entries directly in their native local currency units (e.g., INR for Indian markets). "
            "   When displaying or graphing comparisons, cleanly state the currencies being evaluated so values match contextually.\n\n"
            "CRITICAL VISUALIZATION RULE:\n"
            "If the question evaluates or compares metrics across multiple years or companies, append a raw comma-separated dataset "
            "block at the very end of your response on a brand new line starting strictly with 'DATA_SERIES:' followed by pairs of Year_Company=Value.\n"
        )
        system_instruction += hybrid_rules
        
        past_memory_tickets = self.format_chat_history()
        api_messages = [{"role": "system", "content": system_instruction}] + past_memory_tickets
        api_messages[-1] = {"role": "user", "content": user_payload}
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
                temperature=0.0
            )
            ai_response = response.choices[0].message.content
            
            # Formatting cleanups for display handling
            ai_response = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', ai_response, flags=re.DOTALL)
            ai_response = re.sub(r'\\\[\s*(.*?)\s*\]', r'$$\1$$', ai_response, flags=re.DOTALL)
            ai_response = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', ai_response, flags=re.DOTALL)
            
            self.conversation_history.append({'speaker': 'assistant', 'text': ai_response})
            return ai_response
        except Exception as e:
            return f"An operational error occurred: {str(e)}"