import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Import your existing RAG retrieval components
from tools.vector_db import query_vector_db
from pipelines.rag_pipeline import format_context

load_dotenv()
client = OpenAI()

# 🏆 THE GOLDEN TEST SET (Questions, Context Triggers, and Expected Ground Truths)
EVAL_DATASET = [
    {
        "query": "What was the revenue of amex in 2023?",
        "expected_ground_truth": "American Express (Amex) reported a total consolidated revenue of $60,515 million ($60.5B) for the full year 2023.",
        "requires_web": False
    },
    {
        "query": "What was the total revenue of axis bank?",
        "expected_ground_truth": "Axis Bank's financial metrics require live market data retrieval from external exchange feeds.",
        "requires_web": True
    }
]

def run_isolated_rag_pipeline(user_query):
    """Bypasses Celery/UI to simulate the core RAG extraction logic synchronously."""
    # 1. Retrieve raw contexts from local Chroma DB
    docs, metas = query_vector_db(user_query, n_results=4)
    local_context = format_context((docs, metas)) if docs else "No document snippets found."
    
    # 2. Simulate final generation payload synthesis
    system_prompt = "You are a financial analyst. Answer the question based strictly on the context provided."
    user_payload = f"Context:\n{local_context}\n\nQuestion: {user_query}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content, local_context
    except Exception as e:
        return f"Generation Error: {str(e)}", local_context

def evaluate_faithfulness(generated_answer, retrieved_context):
    """Quantifies Faithfulness: Are the claims supported by the context?"""
    prompt = f"""
    Analyze the Generated Answer against the True Context.
    Step 1: Break down the Generated Answer into discrete factual claims.
    Step 2: For each claim, check if it is directly supported by the True Context (Yes or No).
    
    Output exactly in this format:
    Total Claims: [integer]
    Supported Claims: [integer]
    
    True Context: {retrieved_context}
    Generated Answer: {generated_answer}
    """
    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0)
        text = res.choices[0].message.content
        total = int(re.search(r"Total Claims:\s*(\d+)", text).group(1))
        supported = int(re.search(r"Supported Claims:\s*(\d+)", text).group(1))
        return round(supported / total, 2) if total > 0 else 1.0
    except:
        return 0.0

def evaluate_answer_relevance(query, generated_answer):
    """Quantifies Answer Relevance using algorithmic semantic overlap mapping."""
    prompt = f"""
    Rate how directly the Generated Answer addresses the User Query. 
    Ignore factual truth—judge only if it directly answers the question without rambling or adding irrelevant disclosures.
    Output a single float score strictly between 0.0 (completely irrelevant) and 1.0 (perfectly relevant).
    
    User Query: {query}
    Generated Answer: {generated_answer}
    Score:"""
    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return float(re.search(r"(\d\.\d+)", res.choices[0].message.content).group(1))
    except:
        return 0.0

def evaluate_context_recall(retrieved_context, ground_truth):
    """Quantifies Context Recall: Did our vector search fetch the necessary target facts?"""
    prompt = f"""
    Compare the Retrieved Context against the Ground Truth statement.
    Does the Retrieved Context contain the core financial facts mentioned in the Ground Truth?
    Output exactly:
    Found: Yes (or Found: No)
    
    Ground Truth: {ground_truth}
    Retrieved Context: {retrieved_context}
    """
    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return 1.0 if "Found: Yes" in res.choices[0].message.content else 0.0
    except:
        return 0.0

if __name__ == "__main__":
    print("\n🚀 STARTING DIRECT PROGRAMMATIC RAG METRICS AUDIT...\n" + "="*60)
    
    for idx, test_case in enumerate(EVAL_DATASET, 1):
        query = test_case["query"]
        truth = test_case["expected_ground_truth"]
        
        print(f"\n📝 TEST CASE #{idx} | Query: '{query}'")
        
        # Run synchronous RAG execution loop
        answer, context = run_isolated_rag_pipeline(query)
        
        # Calculate metric values dynamically via LLM-as-a-judge
        faithfulness = evaluate_faithfulness(answer, context)
        relevance = evaluate_answer_relevance(query, answer)
        recall = evaluate_context_recall(context, truth)
        
        print("-" * 40)
        print(f"🤖 Agent Output Summary:\n{answer[:120]}...")
        print("-" * 40)
        print(f"📊 QUANTIFIED METRICS MATRIX:")
        print(f"   🔹 Faithfulness (Hallucination Guardrail) : {int(faithfulness * 100)}%")
        print(f"   🔹 Answer Relevance (Intent Drift)       : {int(relevance * 100)}%")
        print(f"   🔹 Context Recall (Retrieval Coverage)     : {int(recall * 100)}%")
        print("="*60)