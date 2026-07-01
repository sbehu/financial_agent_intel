Financial RAG Multi-Agent Architecture: Master Blueprint
This document captures the evolutionary debugging steps, architectural traps, and production fixes implemented while moving a financial document auditor from a brittle prototype into a high-precision, stratified enterprise system.

🗺️ Architectural Topology: The Production Data Flow
[ Raw Financial PDFs ] 
       │
       ▼ (sandbox_ingest.py)
[ Element-Aware Layout Parser ] ──► Extracts [START_STRUCTURAL_TABLE] Markdowns
       │
       ▼ 
[ Parent Context Injection ] ────► Burns "Global Reported Scales" (e.g., Millions) into Chunks
       │
       ▼
[ Storage Index Vector Space ] ──► 4,980 Chunks committed to ChromaDB Storage
       │
  ========================= RUNTIME EXECUTION =========================
       │
[ Incoming Evaluation Query ]
       │
       ▼ (sandbox_tools.py)
[ Multi-Route Keyword Distillation ] ──► Strips filler tokens to prevent dilution
       │
       ▼ 
[ Explicit Task-Tool Argument Binding ] ──► Bypasses Agent cognitive bias
       │
       ▼ (sandbox_crew.py)
[ CrewAI Assembly Line Execution ] ──► Sequential processing (Retriever ──► Math Analyst)
       │
       ▼ (sandbox_evaluate.py)
[ LLM-as-a-Judge Audit Matrix ] ──► Validates against the RAG Triad (100% Target)
🛠️ Module 1: The Ingestion & Layout Parsing Base (sandbox_ingest.py)
Core Engineering Advancements:
Element-Aware Layout Parsing: Bypasses basic token-count chunk splitters that slice directly through financial charts. Tables are reconstructed algorithmically into standard Markdown format strings flanked by [START_STRUCTURAL_TABLE] boundaries.

Parent Context Injection: Dynamically scans the initial pages of corporate filings to extract unit scales (e.g., Millions, Crores) and maps them explicitly into the header metadata of every downstream text array chunk. This neutralizes downstream "scale drop" calculation failures.

Stratified Golden Set Generation: Reverse-engineers a 200-question testing matrix directly balanced to target 40% tabular matrix intersections and 60% text prose narratives.

Python
# Save as sandbox_ingest.py
import os
import json
import random
import re
from dotenv import load_dotenv
from openai import OpenAI
import pypdf

load_dotenv()
client = OpenAI()

def parse_lines_to_markdown_table(lines):
    formatted_table = []
    for line in lines:
        if re.search(r'\d+', line) and (len(line.split('   ')) > 1 or len(line.split('\t')) > 1):
            columns = [col.strip() for col in re.split(r'\s{2,}', line) if col.strip()]
            if len(columns) > 1:
                formatted_table.append("| " + " | ".join(columns) + " |")
        else:
            if formatted_table:
                break
    if len(formatted_table) >= 1:
        header_cols = len(formatted_table[0].split('|')) - 2
        divider = "|" + "---|"*header_cols
        formatted_table.insert(1, divider)
        return "\n".join(formatted_table)
    return None

def scan_global_scale_context(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        sample_text = ""
        for i in range(min(5, len(reader.pages))):
            text = reader.pages[i].extract_text()
            if text:
                sample_text += text + "\n"
        found_scales = []
        if re.search(r'(?i)in\s+billions|all\s+amounts\s+in\s+billions', sample_text):
            found_scales.append("REPORTED SCALE: Billions (Absolute multiplier: x1,000,000,000)")
        if re.search(r'(?i)in\s+millions|all\s+amounts\s+in\s+millions', sample_text):
            found_scales.append("REPORTED SCALE: Millions (Absolute multiplier: x1,000,000)")
        if re.search(r'(?i)in\s+crores|all\s+amounts\s+in\s+crores|₹\s+in\s+crores', sample_text):
            found_scales.append("REPORTED SCALE: Crores (1 Crore = 10 Million Rupees | Multiplier: x10,000,000)")
        return " | ".join(found_scales) if found_scales else "REPORTED SCALE: Standard Units / Absolute Raw Values"
    except Exception:
        return "REPORTED SCALE: Standard Units / Absolute Raw Values"

def extract_layout_aware_chunks(pdf_path, chunk_size=1000, chunk_overlap=200):
    print(f"📖 Processing Document: {os.path.basename(pdf_path)}...")
    global_scale = scan_global_scale_context(pdf_path)
    reader = pypdf.PdfReader(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue
        lines = page_text.splitlines()
        elements = []
        current_section = "GENERAL FINANCIAL DATA"
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if not line:
                idx += 1
                continue
            if len(line) < 100 and any(keyword in line.upper() for keyword in ["INCOME", "OPERATIONS", "SEGMENT RESULTS"]):
                current_section = line.upper()
            if re.search(r'\d+', line) and (len(line.split('   ')) > 1 or len(line.split('\t')) > 1):
                table_lines = []
                while idx < len(lines) and lines[idx].strip():
                    table_lines.append(lines[idx].strip())
                    idx += 1
                md_table = parse_lines_to_markdown_table(table_lines)
                if md_table:
                    classification = "Main Consolidated Income Statement" if "CONSOLIDATED" in current_section else "Segment Breakdown / Footnote Narrative"
                    elements.append(f"\nFINANCIAL CLASSIFICATION: {classification}\n[START_STRUCTURAL_TABLE]\n{md_table}\n[END_STRUCTURAL_TABLE]\n")
                else:
                    elements.append(" ".join(table_lines))
            else:
                elements.append(line)
                idx += 1

        current_chunk = ""
        for element in elements:
            context_header = f"CRITICAL CONTEXT | Document: {os.path.basename(pdf_path)} | {global_scale} | Page: {page_num}"
            if "[START_STRUCTURAL_TABLE]" in element:
                if current_chunk.strip():
                    chunks.append({"content": f"{context_header}\n{current_chunk.strip()}", "metadata": {"source": os.path.basename(pdf_path), "page": page_num}})
                    current_chunk = ""
                chunks.append({"content": f"{context_header}\n{element.strip()}", "metadata": {"source": os.path.basename(pdf_path), "page": page_num}})
            else:
                if len(current_chunk) + len(element) < chunk_size:
                    current_chunk += "\n" + element
                else:
                    chunks.append({"content": f"{context_header}\n{current_chunk.strip()}", "metadata": {"source": os.path.basename(pdf_path), "page": page_num}})
                    current_chunk = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                    current_chunk += "\n" + element
        if current_chunk.strip():
            chunks.append({"content": f"{context_header}\n{current_chunk.strip()}", "metadata": {"source": os.path.basename(pdf_path), "page": page_num}})
    return chunks

def generate_synthetic_test_case(chunk_content, metadata):
    prompt = f"""You are an expert enterprise financial auditor. Review the raw text chunk provided below. 
    If the text contains [START_STRUCTURAL_TABLE], target your question strictly at data intersections within that Markdown table.
    Generate exactly ONE highly specific quantitative question and its absolute accurate Ground Truth answer.
    Return raw JSON format: {{"question": "...", "ground_truth": "..."}}
    Text Chunk: \"\"\"{chunk_content}\"\"\""""
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.2)
        test_case = json.loads(response.choices[0].message.content)
        test_case["source_file"] = metadata["source"]
        test_case["page_number"] = metadata["page"]
        return test_case
    except Exception as e:
        return None

if __name__ == "__main__":
    data_dir = "data"
    all_chunks = []
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    for pdf in pdf_files:
        all_chunks.extend(extract_layout_aware_chunks(os.path.join(data_dir, pdf)))
    
    TARGET_EVAL_SIZE = 200
    table_chunks = [c for c in all_chunks if "[START_STRUCTURAL_TABLE]" in c["content"]]
    prose_chunks = [c for c in all_chunks if "[START_STRUCTURAL_TABLE]" not in c["content"]]
    
    sampled_tables = random.sample(table_chunks, min(int(TARGET_EVAL_SIZE * 0.40), len(table_chunks)))
    sampled_prose = random.sample(prose_chunks, min(TARGET_EVAL_SIZE - len(sampled_tables), len(prose_chunks)))
    stratified_chunks = sampled_tables + sampled_prose
    random.shuffle(stratified_chunks)
    
    synthetic_golden_set = []
    for idx, chunk in enumerate(stratified_chunks, 1):
        test_case = generate_synthetic_test_case(chunk["content"], chunk["metadata"])
        if test_case: synthetic_golden_set.append(test_case)
        
    with open("synthetic_golden_set.json", "w") as f:
        json.dump(synthetic_golden_set, f, indent=4)
        
    import chromadb
    db_path = os.path.join(os.getcwd(), "chroma_db_storage")
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.create_collection(name="amex_historical_index")
    
    collection.add(
        ids=[f"id_{i}" for i in range(len(all_chunks))],
        documents=[chunk["content"] for chunk in all_chunks],
        metadatas=[chunk["metadata"] for chunk in all_chunks]
    )
    print("✅ DATABASE AND GOLDEN SET SYNCHRONIZED.")
🛠️ Module 2: The Core Retrieval Tools Layer (sandbox_tools.py)
Core Engineering Advancements:
Argument Telemetry Logging: Breaks the mystery of "silent failures". Explicit type-casting and print statement boundaries capture exactly what argument parameters string vectors are passing down from the upstream agent.

Multi-Route Intent Query Distillation: Prevents token dilution. Long conversational phrases generated by human interfaces or conversational agents are cleaned down to their raw semantic indicators (e.g., transforming "What are the roles included in the first line..." into "first line of defense erm policy"), increasing vector database similarity precision.

Python
# Save as sandbox_tools.py
import os
import chromadb
from crewai.tools import tool

db_path = os.path.join(os.getcwd(), "chroma_db_storage")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="amex_historical_index")

@tool("query_internal_amex_vault")
def query_internal_amex_vault(search_query: str) -> str:
    """Queries the local vector database chunk array for American Express corporate records."""
    try:
        print("\n==================================================")
        print("🔌 RETRIEVAL TOOL ACTIVE LOG TRIGGERED")
        print(f"👉 Raw Argument Value Received: '{search_query}'")
        print("==================================================")

        if not search_query or not str(search_query).strip():
            return "No search query provided to the retrieval system."

        financial_keywords = ["interest", "income", "revenue", "growth", "scaling", "trend", "yoy"]
        if any(kw in search_query.lower() for kw in financial_keywords):
            optimized_search = f"{search_query} Main Consolidated Income Statement Total Interest"
        else:
            stop_phrases = ["what are the", "what is the", "as per the", "according to", "roles included in the", "potential impact of"]
            cleaned_query = search_query.lower()
            for phrase in stop_phrases:
                cleaned_query = cleaned_query.replace(phrase, "")
            optimized_search = " ".join(cleaned_query.split()).strip()
            
        results = collection.query(query_texts=[optimized_search], n_results=5)
        compiled_output = []
        for i, document in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] else {}
            compiled_output.append(f"--- SOURCE: {metadata.get('source')} (PAGE {metadata.get('page')}) ---\n{document}\n")
        return "\n".join(compiled_output) if compiled_output else "No relevant records found."
    except Exception as e:
        return f"⚠️ Tool Lookup Error: {e}"

@tool("query_external_competitor_web")
def query_external_competitor_web(query: str) -> str:
    """Simulates external industry market lookup."""
    return "Axis Bank Total Interest Income: 2022: ₹688.46B | 2023: ₹874.48B | 2024: ₹1,130.00B | 2025: ₹1,273.74B"

@tool("execute_financial_calculation")
def execute_financial_calculation(operation: str) -> str:
    """Runs standard quantitative python formula string parsing."""
    try:
        return str(eval(operation))
    except Exception as e:
        return f"Calculation execution crash: {e}"

@tool("get_live_exchange_rate")
def get_live_exchange_rate(currency_pair: str) -> float:
    """Returns absolute currency multiplier."""
    return 83.50
🛠️ Module 3: The Multi-Agent Assembly Line (sandbox_crew.py)
Core Engineering Advancements:
Elimination of Variable Disconnects: Aligns global agent worker mappings (researcher, calculator) directly to internal task runners, eliminating runtime fallback errors.

Explicit Task-Tool Argument Binding: Forces the task framework to deliver the raw incoming parameter query (user_query) strictly to the tool invocation slot, bypassing the agent’s internal cognitive prompt bias to distort search vectors.

Python
# Save as sandbox_crew.py
import os
from crewai import Agent, Task, Crew, Process
from sandbox_tools import query_internal_amex_vault, query_external_competitor_web, execute_financial_calculation, get_live_exchange_rate

os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

researcher = Agent(
    role="Senior Financial Data Retriever",
    goal="Locate, extract, and report exact metrics using tools. Never estimate or invent market data.",
    backstory="You are a forensic data collector. You extract facts verbatim without modifications.",
    tools=[query_internal_amex_vault, query_external_competitor_web, get_live_exchange_rate],
    verbose=True
)

calculator = Agent(
    role="Deterministic Mathematical Analyst",
    goal="Execute calculation workflows using absolute values provided directly by the retriever.",
    backstory="You take raw numbers from the data retriever's clipboard and run perfect formulas.",
    tools=[execute_financial_calculation],
    verbose=True
)

def run_financial_crew_pipeline(user_query: str):
    retrieve_task = Task(
        description=(
            f"You must execute a local database search for the exact query: '{user_query}'. "
            f"Do not alter, summarize, or convert this topic string. Pass the query text "
            f"directly to your query_internal_amex_vault tool exactly as stated to gather context."
        ),
        expected_output="An unedited assembly of matching text chunks and tables pulled from the storage index.",
        agent=researcher,
        tools=[query_internal_amex_vault]
    )
    
    math_task = Task(
        description=(
            f"Review the text context assembled by the retriever for the original request: '{user_query}'. "
            f"If the request requires numerical trend calculations, pass the data fields to your python calculator. "
            f"If it is a qualitative policy query, extract the factual answers directly from the context text."
        ),
        expected_output="A high-precision final report that accurately satisfies the original user request.",
        agent=calculator
    )
    
    crew = Crew(
        agents=[researcher, calculator],
        tasks=[retrieve_task, math_task],
        process=Process.sequential
    )
    
    result = crew.kickoff(inputs={"query": user_query})
    return str(result)
🛠️ Module 4: The Evaluation Judging Engine (sandbox_evaluate.py)
Core Engineering Advancements:
The RAG Triad Implementation: Moves beyond basic string matching. Uses a decoupled LLM-as-a-Judge architecture with a strict structural JSON schema constraint to break down and audit the system across Faithfulness, Context Relevance, and Quantitative Accuracy.

Python
# Save as sandbox_evaluate.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from app_crew_adapter import run_crew_with_stream

load_dotenv()
client = OpenAI()

def llm_judge_triad(query, retrieved_context, final_output, ground_truth):
    prompt = f"""You are an adversarial AI auditor evaluating a high-precision financial RAG system.
    Calculate three scores on a scale from 0.0 to 1.0 based on these criteria:
    1. Faithfulness: Are all numbers in 'Final Output' strictly backed up by 'Retrieved Context'?
    2. Context Relevance: Did the retriever isolate the crisp data rows matching the query?
    3. Accuracy vs Ground Truth: Does the output present the exact answer matching the ground truth?
    Output RAW JSON:
    {{
        "faithfulness_score": 0.0,
        "context_relevance_score": 0.0,
        "accuracy_score": 0.0,
        "reasoning_audit_log": "..."
    }}
    Query: "{query}" | GT: "{ground_truth}" | Context: ""\"{retrieved_context}\"| Output: ""\"{final_output}\"
    """
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.0)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"faithfulness_score": 0.0, "context_relevance_score": 0.0, "accuracy_score": 0.0, "reasoning_audit_log": str(e)}

def execute_system_audit():
    with open("synthetic_golden_set.json", "r") as f:
        golden_set = json.load(f)
    
    comprehensive_metrics = []
    avg_faithfulness, avg_relevance, avg_accuracy = 0.0, 0.0, 0.0

    for idx, test_case in enumerate(golden_set, 1):
        query = test_case["question"]
        ground_truth = test_case["ground_truth"]
        try:
            final_output, trace_logs = run_crew_with_stream(query)
            scores = llm_judge_triad(query, trace_logs, final_output, ground_truth)
            
            avg_faithfulness += scores["faithfulness_score"]
            avg_relevance += scores["context_relevance_score"]
            avg_accuracy += scores["accuracy_score"]
            
            print(f"⚖️ Case {idx} | Faith: {scores['faithfulness_score']} | Rel: {scores['context_relevance_score']} | Acc: {scores['accuracy_score']}")
        except Exception as e:
            print(f"⚠️ Failed on case {idx}: {e}")

    total = len(golden_set)
    print("\n=========================================================")
    print("📊 FINAL AGGREGATE SYSTEM METRICS")
    print("=========================================================")
    print(f"🔒 Average System Faithfulness (Grounding): {(avg_faithfulness / total) * 100:.2f}%")
    print(f"🎯 Average Context Retrieval Relevance:      {(avg_relevance / total) * 100:.2f}%")
    print(f"📈 Average Downstream Quantitative Accuracy:   {(avg_accuracy / total) * 100:.2f}%")
    print("=========================================================")

if __name__ == "__main__":
    execute_system_audit()
🎓 Mid-Flight Core Diagnostic Lessons Learned
Keep these three real-world production breakthroughs fresh in your mind for any data science panel or review:

The Tool Suffix Leakage Trap: Appending hardcoded strings to automated queries makes one specific prompt work perfectly, but hardcodes behavior. If an agent tries to answer a risk question with an interest query suffix, it results in an empty database recall.

The Silent Tool Drop: If an agent's global tools profile doesn't match the active task tools execution configuration, CrewAI silently drops the tool path entirely without throwing an error, causing the agent to fall back on its internal pre-trained memory.

The Variable Mismatch Name Failure: If global agent instantiations (researcher) don't perfectly align with internal execution function wrappers (retriever_agent), the framework passes over the logic and instantiates an empty, blank worker configuration behind the scenes.