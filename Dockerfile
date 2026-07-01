FROM chromadb/chroma:0.4.24

# Set up Python environment inside the pre-baked container
RUN apt-get update && apt-get install -y python3-pip python3-dev

WORKDIR /financial_agent_intel

# Copy requirements
COPY requirements.txt .

# Install only the remaining lightweight tools
RUN pip3 install --no-cache-dir -r requirements.txt

# 🌟 STEP 1: Copy your local database folder explicitly into the correct workspace path first
#COPY chroma_db_storage/ /financial_agent_intel/chroma_db_storage/
# 🌟 FIX: Create a fresh, clean database directory dynamically inside the cloud container

RUN mkdir -p /financial_agent_intel/chroma_db_storage

# 🌟 STEP 2: Now copy the rest of your files safely around it
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "financial_analyst_app.py", "--server.port=8501", "--server.address=0.0.0.0"]