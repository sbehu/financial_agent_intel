import os
from celery import Celery
from utils.logger import setup_custom_logger
from agents.financial_agent import FinancialAgent

logger = setup_custom_logger("CeleryWorker")

# 📬 Connect Celery to our Redis box
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("financial_tasks", broker=REDIS_URL, backend=REDIS_URL)

# 💼 Initialize the heavy AI Agent inside the worker memory space
financial_agent = FinancialAgent()

@celery_app.task(name="tasks.process_document_pipeline")
def process_document_pipeline(user_query):
    """
    Background worker task executing the actual agent query loop 
    safely out of sight of the main web screen.
    """
    logger.info(f"🏁 Task Received: Processing query: '{user_query}'")
    
    # 🧠 Calling the synchronized master processing orchestrator function
    response = financial_agent.process_query(user_query)
    
    logger.info(f"✅ Task Completed successfully!")
    return response