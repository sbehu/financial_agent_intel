import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from app_crew_adapter import run_crew_with_stream  # Your system's execution pipeline

# Load environment variables
load_dotenv()
client = OpenAI()

def llm_judge_triad(query, retrieved_context, final_output, ground_truth):
    """
    Production LLM-as-a-Judge engine that evaluates the system output 
    across the standard RAG Triad using explicit fractional scoring rules.
    """
    prompt = f"""
    You are an independent, adversarial AI quality auditor evaluating a high-precision financial RAG system.
    Review the audit artifacts provided below and calculate three distinct scores on a scale from 0.0 (Worst) to 1.0 (Perfect).
    
    EVALUATION MATRIX CRITERIA:
    1. Faithfulness (Grounding): Are all numbers and claims in the 'Final Output' strictly backed up by the 'Retrieved Context'? 
       Decline points if the output introduces outside metrics or hallucinations not contained in the context text.
    
    2. Context Relevance: How well did the retriever isolate the exact table rows matching the query? 
       Give 1.0 if the context contains the crisp markdown table matching the exact target metrics. Give 0.0 if it is full of unrelated text prose.
       
    3. Accuracy vs Ground Truth: Does the 'Final Output' successfully present the exact numbers or trends requested, matching the 'Target Ground Truth'?

    CRITICAL: You must output your analysis in raw JSON format exactly matching this structure:
    {{
        "faithfulness_score": 0.00,
        "context_relevance_score": 0.00,
        "accuracy_score": 0.00,
        "reasoning_audit_log": "A brief, highly technical explanation of the scores assigned."
    }}

    AUDIT ARTIFACTS:
    - User Original Query: "{query}"
    - Target Ground Truth Answer: "{ground_truth}"
    - System Retrieved Context: \"\"\"{retrieved_context}\"\"\"
    - System Final Output: \"\"\"{final_output}\"\"\"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "faithfulness_score": 0.0,
            "context_relevance_score": 0.0,
            "accuracy_score": 0.0,
            "reasoning_audit_log": f"Judge failed to complete calculation: {e}"
        }

def execute_system_audit():
    golden_set_path = "synthetic_golden_set.json"
    
    if not os.path.exists(golden_set_path):
        print(f"❌ EVALUATION ERROR: Golden dataset file '{golden_set_path}' is missing.")
        print("👉 Please run your sandbox_ingest.py pipeline first to auto-generate validation vectors.")
        return

    with open(golden_set_path, "r") as f:
        golden_set = json.load(f)

    print(f"🚀 Financial RAG Automated Evaluator Initialized.")
    print(f"📊 Processing {len(golden_set)} validated test vectors against active Crew structures...\n")
    
    comprehensive_metrics = []
    
    # Aggregators for system averages
    avg_faithfulness = 0.0
    avg_relevance = 0.0
    avg_accuracy = 0.0

    for idx, test_case in enumerate(golden_set, 1):
        query = test_case["question"]
        ground_truth = test_case["ground_truth"]
        
        print(f"📋 [Test Case {idx}/{len(golden_set)}]")
        print(f"   ❓ Query: '{query}'")
        print(f"   🎯 Target: '{ground_truth}'")
        
        try:
            # 1. Fire the actual live system multi-agent assembly line
            final_output, trace_logs = run_crew_with_stream(query)
            
            # 2. Pass the pipeline outputs to the LLM judge engine
            scores = llm_judge_triad(query, trace_logs, final_output, ground_truth)
            
            # Accumulate scores
            avg_faithfulness += scores["faithfulness_score"]
            avg_relevance += scores["context_relevance_score"]
            avg_accuracy += scores["accuracy_score"]
            
            # Store detail matrix block
            comprehensive_metrics.append({
                "test_case_id": idx,
                "query": query,
                "ground_truth": ground_truth,
                "pipeline_response": final_output,
                "scores": scores
            })
            
            print(f"   ⚖️  Judge Scores ➔ Faithfulness: {scores['faithfulness_score']} | Relevance: {scores['context_relevance_score']} | Accuracy: {scores['accuracy_score']}")
            print(f"   📝 Log: {scores['reasoning_audit_log']}\n")
            
        except Exception as e:
            print(f"   ⚠️ System aborted on row execution loop: {e}\n")

    # Final Summary Matrix Calculations
    total_cases = len(golden_set)
    if total_cases > 0:
        print("=========================================================")
        print("📊 FINAL AGGREGATE SYSTEM METRICS")
        print("=========================================================")
        print(f"🔒 Average System Faithfulness (Grounding): {(avg_faithfulness / total_cases) * 100:.2f}%")
        print(f"🎯 Average Context Retrieval Relevance:      {(avg_relevance / total_cases) * 100:.2f}%")
        print(f"📈 Average Downstream Quantitative Accuracy:   {(avg_accuracy / total_cases) * 100:.2f}%")
        print("=========================================================")
        
        # Write clean audit telemetry log to root path
        with open("production_evaluation_report.json", "w") as f:
            json.dump(comprehensive_metrics, f, indent=4)
        print("\n💾 Matrix telemetry log securely written to 'production_evaluation_report.json'.")

if __name__ == "__main__":
    execute_system_audit()