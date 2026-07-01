import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Import the clean, decorated tools directly from our tools backend
from sandbox_tools import (
    query_internal_amex_vault, 
    query_external_competitor_web, 
    execute_financial_calculation,
    get_live_exchange_rate
)

# Set the global model environment variable CrewAI uses natively under the hood
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

# 🌟 FIXED: Explicit LangChain Model Definitions to align with explicit object properties
production_llm_main = ChatOpenAI(
    model="gpt-4o",  
    temperature=0.2
)

production_llm_mini = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0.2
)

# ──── STEP 1: DEFINE THE SPECIALIZED WORKERS ────

researcher = Agent(
    role="Senior Financial Data Retriever",
    goal="Locate, extract, and report exact metrics using tools. Never estimate or invent market data.",
    backstory=(
        "You are a forensic data collector. When a currency lookup tool returns a value like 94.3, "
        "you must use that exact number. You are strictly forbidden from altering values, applying "
        "historical baseline guesses (like 83.5), or rounding numbers before handing them to the next task."
    ),
    tools=[query_internal_amex_vault, query_external_competitor_web, get_live_exchange_rate],
    llm=production_llm_mini, # 🌟 FIXED: Passing the actual instantiated model object instance
    verbose=True
)

calculator = Agent(
    role="Deterministic Mathematical Analyst",
    goal="Execute calculation workflows using absolute values provided directly by the retriever.",
    backstory=(
        "You take raw numbers from the data retriever's clipboard. If the retriever gives you an exchange rate "
        "of 94.3, you must use 94.3. You must convert all unit expressions (like 'billions' or 'crores') "
        "into absolute digits (e.g., writing out all the zeros) before running calculations to prevent scale errors."
    ),
    tools=[execute_financial_calculation],
    llm=production_llm_mini, # 🌟 FIXED: Passing the actual instantiated model object instance
    verbose=True
)

# ──── STEP 2: CREW EXECUTION ENTRYPOINT ────

def run_financial_crew_pipeline(user_query: str):
    """Executes the multi-agent assembly line with rigid argument safety gates."""
    
    # 2. Update your retrieve task to explicitly control how arguments flow
    # 🌟 FIXED: Mapped agent parameter directly to the global 'researcher' object
    retrieve_task = Task(
        description=(
            f"You must execute a local database search for the exact query: '{user_query}'. "
            f"Do not alter, summarize, or convert this topic string. Pass the query text "
            f"directly to your retrieval tools to gather background context. If a currency "
            f"conversion is required by the query metrics, dynamically utilize your live exchange rate tool."
        ),
        expected_output="An unedited assembly of matching text chunks, tables, and calculated value contexts.",
        agent=researcher,
        # 🌟 THE SYNC FIX: Explicitly grant access to ALL required search/conversion tools 
        # so the framework never triggers a Silent Tool Drop under heavy load.
        tools=[query_internal_amex_vault, query_external_competitor_web, get_live_exchange_rate]
    )
    
    # 🌟 FIXED: Mapped agent parameter directly to the global 'calculator' object
    math_task = Task(
        description=(
            f"Review the text context assembled by the retriever for the original request: '{user_query}'. "
            f"If the request requires numerical trend calculations or YoY growth metrics, pass the data fields "
            f"to your python math calculator. If it is a qualitative policy or risk management query, "
            f"extract the factual answers directly from the context text matching the ground truth."
        ),
        expected_output="A high-precision final report that accurately satisfies the original user request.",
        agent=calculator
    )
    
    # 3. Initialize and kickoff your multi-agent architecture execution
    # 🌟 FIXED: Passing aligned agents list matching Step 1 variables
    crew = Crew(
        agents=[researcher, calculator],
        tasks=[retrieve_task, math_task],
        process=Process.sequential
    )
    
    # Pass the incoming user string down to the execution runtime engine
    result = crew.kickoff(inputs={"query": user_query})
    return str(result)