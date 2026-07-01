import os
import json
import random
import re
from dotenv import load_dotenv
from openai import OpenAI
import pypdf

# Load environment variables
load_dotenv()
client = OpenAI()

def parse_lines_to_markdown_table(lines):
    """
    Algorithmic helper that detects financial table structures within plain text 
    and converts them into standard, structural Markdown tables.
    """
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
    """
    Scans the initial pages of a financial report to dynamically harvest
    the absolute scale context (e.g., Millions, Billions, Crores).
    """
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
    """
    Reads a PDF page-by-page, extracts textual arrays, algorithmically isolates
    structural tables, tracks section headers, and injects parent document scale 
    and structural classification tokens into every block element.
    """
    print(f"📖 Processing Document: {os.path.basename(pdf_path)}...")
    
    global_scale = scan_global_scale_context(pdf_path)
    print(f"   🎯 Global Context Captured ➔ Context Token: [{global_scale}]")
    
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
                
            if len(line) < 100 and any(keyword in line.upper() for keyword in ["CONSOLIDATED STATEMENTS OF INCOME", "CONSOLIDATED STATEMENT OF INCOME", "STATEMENT OF OPERATIONS", "SEGMENT RESULTS", "INCOME STATEMENTS"]):
                current_section = line.upper()
                
            if re.search(r'\d+', line) and (len(line.split('   ')) > 1 or len(line.split('\t')) > 1):
                table_lines = []
                while idx < len(lines) and lines[idx].strip():
                    table_lines.append(lines[idx].strip())
                    idx += 1
                
                md_table = parse_lines_to_markdown_table(table_lines)
                if md_table:
                    if "CONSOLIDATED" in current_section and any(k in current_section for k in ["INCOME", "OPERATIONS"]):
                        classification = "Main Consolidated Income Statement"
                    else:
                        classification = "Segment Breakdown / Footnote Narrative"
                        
                    classification_header = f"FINANCIAL CLASSIFICATION: {classification}\n"
                    elements.append(f"\n{classification_header}[START_STRUCTURAL_TABLE]\n{md_table}\n[END_STRUCTURAL_TABLE]\n")
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
                    chunks.append({
                        "content": f"{context_header}\n{current_chunk.strip()}",
                        "metadata": {"source": os.path.basename(pdf_path), "page": page_num}
                    })
                    current_chunk = ""
                chunks.append({
                    "content": f"{context_header}\n{element.strip()}",
                    "metadata": {"source": os.path.basename(pdf_path), "page": page_num}
                })
            else:
                if len(current_chunk) + len(element) < chunk_size:
                    current_chunk += "\n" + element
                else:
                    chunks.append({
                        "content": f"{context_header}\n{current_chunk.strip()}",
                        "metadata": {"source": os.path.basename(pdf_path), "page": page_num}
                    })
                    current_chunk = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                    current_chunk += "\n" + element

        if current_chunk.strip():
            chunks.append({
                "content": f"{context_header}\n{current_chunk.strip()}",
                "metadata": {"source": os.path.basename(pdf_path), "page": page_num}
            })
            
    print(f"   └─ Extracted {len(chunks)} structural table-protected chunks.")
    return chunks

def generate_synthetic_test_case(chunk_content, metadata):
    """
    Leverages an LLM to automatically synthesize a high-quality QA validation row.
    """
    prompt = f"""
    You are an expert enterprise financial auditor evaluating historical compliance reports.
    Review the raw document text chunk provided below. 
    
    If the text chunk contains a section marked with [START_STRUCTURAL_TABLE], you MUST target your question 
    and answer strictly at the data intersections or comparative metrics within that Markdown table.
    
    Generate exactly ONE highly specific quantitative question and its absolute accurate Ground Truth answer.
    
    CRITICAL STRUCTURE: You must output your response in raw JSON format exactly like this:
    {{
        "question": "The specific quantitative or matrix-intersection question.",
        "ground_truth": "The exact precise numeric or factual answer."
    }}

    Text Chunk:
    \"\"\"{chunk_content}\"\"\"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        test_case = json.loads(response.choices[0].message.content)
        test_case["source_file"] = metadata["source"]
        test_case["page_number"] = metadata["page"]
        return test_case
    except Exception as e:
        print(f"⚠️ Failed to generate test case for a chunk: {e}")
        return None

if __name__ == "__main__":
    print("\n🚀 Sandbox Ingestion Environment Initialized.")
    
    data_dir = "data"
    all_chunks = []
    
    if not os.path.exists(data_dir):
        print(f"❌ DIRECTORY ERROR: Target folder '{data_dir}/' does not exist at your project root.")
        print(f"👉 Resolution: Please create a directory named '{data_dir}' in c:\\financial_agent_intel\\")
    else:
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"⚠️ EMPTY FOLDER: No PDF files found inside the '{data_dir}/' directory.")
            print("👉 Resolution: Paste your target financial PDF reports inside that folder and re-run.")
        else:
            print(f"📂 Found {len(pdf_files)} target file(s) inside directory layer.")
            for pdf in pdf_files:
                pdf_full_path = os.path.join(data_dir, pdf)
                all_chunks.extend(extract_layout_aware_chunks(pdf_full_path))
                
            print(f"\n📊 Total Ingested Dataset Matrix Size: {len(all_chunks)} chunks.")
            
            if all_chunks:
                # 🌟 PRODUCTION SCALE-UP: Target 200 Stratified Test Cases
                TARGET_EVAL_SIZE = 200
                print(f"\n🧠 Triggering Stratified Data Generation Pipeline (Targeting {TARGET_EVAL_SIZE} cases)...")
                
                # Segregate chunks to guarantee topological diversity
                table_chunks = [c for c in all_chunks if "[START_STRUCTURAL_TABLE]" in c["content"]]
                prose_chunks = [c for c in all_chunks if "[START_STRUCTURAL_TABLE]" not in c["content"]]
                
                print(f"   ├─ Available Table Chunks: {len(table_chunks)}")
                print(f"   └─ Available Prose Chunks: {len(prose_chunks)}")
                
                # Calculate stratified target allocations (40% Tables / 60% Prose & Risk narratives)
                target_tables_count = int(TARGET_EVAL_SIZE * 0.40)
                target_prose_count = TARGET_EVAL_SIZE - target_tables_count
                
                sampled_tables = random.sample(table_chunks, min(target_tables_count, len(table_chunks)))
                sampled_prose = random.sample(prose_chunks, min(target_prose_count, len(prose_chunks)))
                stratified_chunks = sampled_tables + sampled_prose
                random.shuffle(stratified_chunks)  # Interleave the test rows
                
                print(f"\n📈 Stratified Matrix Locked ➔ Selected {len(sampled_tables)} Tables & {len(sampled_prose)} Prose Blocks.")
                
                synthetic_golden_set = []
                for idx, chunk in enumerate(stratified_chunks, 1):
                    print(f"⚙️ Synthesizing row {idx}/{len(stratified_chunks)}...")
                    test_case = generate_synthetic_test_case(chunk["content"], chunk["metadata"])
                    if test_case:
                        synthetic_golden_set.append(test_case)
                
                print("\n🛑 HUMAN-IN-THE-LOOP APPROVAL REQUIRED:")
                print("============================================")
                print(f"Generated {len(synthetic_golden_set)} unique high-precision QA vectors.")
                print("============================================")
                
                approval = input("👉 Do you approve committing this comprehensive matrix to your Golden Set? (yes/no): ")
                
                if approval.strip().lower() == 'yes':
                    output_path = "synthetic_golden_set.json"
                    with open(output_path, "w") as f:
                        json.dump(synthetic_golden_set, f, indent=4)
                    print(f"\n💾 SUCCESS: Saved {len(synthetic_golden_set)} validated rows to '{output_path}'.")

                    print("\n📦 Initializing ChromaDB Storage Engine Injection...")
                    import chromadb
                    
                    db_path = os.path.join(os.getcwd(), "chroma_db_storage")
                    chroma_client = chromadb.PersistentClient(path=db_path)
                    collection_name = "amex_historical_index"
                    
                    try:
                        chroma_client.delete_collection(name=collection_name)
                    except Exception:
                        pass
                        
                    collection = chroma_client.create_collection(name=collection_name)
                    print(f"📥 Loading layout-aware chunks into index matrix '{collection_name}'...")
                    
                    ids = [f"id_{i}" for i in range(len(all_chunks))]
                    documents = [chunk["content"] for chunk in all_chunks]
                    metadatas = [chunk["metadata"] for chunk in all_chunks]
                    
                    batch_size = 400
                    for i in range(0, len(all_chunks), batch_size):
                        end_chunk = min(i + batch_size, len(all_chunks))
                        collection.add(
                            ids=ids[i:end_chunk],
                            documents=documents[i:end_chunk],
                            metadatas=metadatas[i:end_chunk]
                        )
                        print(f"   ⚡ Synced chunks {i} to {end_chunk} successfully.")

                    print(f"\n✅ DATABASE PERSISTENCE SECURED: 200-row testing suite fully prepared.")
                else:
                    print("\n❌ Action aborted by user. Synthetic vectors discarded.")