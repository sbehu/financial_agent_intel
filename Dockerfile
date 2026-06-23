# 🐍 Use the official lightweight Python image
FROM python:3.12-slim

# 🛠️ Set system environment variables to optimize Python inside Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 📁 Create and set the working directory inside the container
WORKDIR /app

# 📦 Install system dependencies needed for compiling certain Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 📃 Copy requirements first to leverage Docker's caching mechanism
COPY requirements.txt .

# ⚡ Install your production-pinned dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# 🚚 Copy the rest of your local application files into the container
COPY . .

# 🔌 Expose the standard Streamlit interface port
EXPOSE 8501

# 🚀 Define the healthcheck to ensure the container is running smoothly
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 🏁 Command to launch your application when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]