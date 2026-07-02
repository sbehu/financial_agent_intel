# AI Financial Analyst Platform: Multi-Agent RAG Pipeline on AWS ECS Fargate
An enterprise-grade, automated AI Financial Analyst Platform engineered to process high-density financial data. The system orchestrates a multi-agent workflow using CrewAI to parse and analyze 6 years of American Express Annual Reports (10-K filings), dynamically benchmarking performance metrics against live competitor data (Visa and Mastercard) fetched via external APIs.

Built with an automated data-science-grade evaluation framework to track system accuracy using the RAG Triad (Faithfulness, Context Relevance, Answer Accuracy).

## 🏗️ Architecture & Component Stack
- **Agent Framework:** CrewAI for task delegation, contextual memory management, and structured agent execution.

- **LLM Base:** gpt-4o-mini utilizing highly restricted system prompts for zero-shot accuracy, specialized math tracking, and evaluation auditing.

- **Vector Database:** ChromaDB configured for persistent local storage, utilizing text-embedding-3-small for semantic representation.

- **Frontend/UI:** Responsive dashboard engineered in Streamlit to surface token consumption, agent trace logs, and comparative visualization charts.

- **Infrastructure & DevOps:** Containerized with Docker, automated via GitHub Actions CI/CD, hosted securely on serverless Amazon ECS Fargate with images securely provisioned in Amazon ECR.

🗺️ Specialized Data Engineering Solutions
1. The Timeline Challenge (Metadata Isolation)
Multi-year financial filings naturally duplicate older baseline numbers for year-over-year context. Naive vector retrievers struggle with temporal blending, pulling fragmented sections from incorrect fiscal cycles. To solve this, this pipeline binds strict fiscal-year metadata tags to database vector chunks to guarantee absolute chronological isolation during the retrieval loop.

2. The Table Parsing Challenge (Structural Chunking & Specialized Agents)
Standard markdown text-splitters frequently slice financial tables in half, severing column headers from key numerical figures and causing downstream LLM hallucinations. This system deploys:

A Structural Chunking Strategy to keep multi-column tabular data blocks integer.

A Data Retrieval Agent assigned exclusively to isolate layout arrays.

A specialized Math Agent operating with background strictures to convert financial string shorthands ("Billions", "M") into absolute, raw digits before computing metrics like asset-to-liability or YoY revenue scaling.

🧪 Automated RAG Triad Evaluation Layer
To systematically measure pipeline behavior without manual "vibe checks," the repository runs an automated Python evaluation testing loop against a Synthetic Golden Dataset of 200+ quantitative/qualitative questions created via stratified sampling across the historical data layout.

The pipeline routes runtime execution traces to an independent Adversarial LLM-as-a-Judge scoring mechanism tracking:

Faithfulness / Grounding: Confirming generated facts are 100% supported by retrieved context nodes.

Context Relevance: Measuring chunk precision against incoming financial inquiries.

Accuracy vs. Ground Truth: Benchmarking calculated numbers against the target golden answer matrix, storing comprehensive telemetry arrays in a telemetry JSON database report.

🚀 Future Scope Roadmap
Hierarchical Parent-Child Chunking: Indexing small token blocks for semantic search while expanding context window footprints (1,000–2,000 tokens) to the LLM to capture table properties fully intact.

Semantic Chunking: Utilizing sentence embedding distances to place chunk dividers dynamically when financial subject boundaries statistically change.

GraphRAG: Injecting a Knowledge Graph framework to cross-link continuous themes and multi-year narrative metrics implicitly.

## 🛠️ Local Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/sbehu/financial_agent_intel.git](https://github.com/sbehu/financial_agent_intel.git)
cd financial_agent_intel


Configure Environment Variables:

Create a .env file in the root directory:
OPENAI_API_KEY=your_openai_api_key_here
CREWAI_LOGGING_LEVEL=INFO


Docker Multi-Container Deployment:

docker build -t financial-agent-app .
docker run -p 8501:8501 financial-agent-app


Open http://localhost:8501 in your browser to interact with the Streamlit engine.