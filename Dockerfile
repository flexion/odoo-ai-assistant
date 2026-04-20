# Dockerfile for Gradio app deployment to AWS ECS Fargate
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-app.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-app.txt && \
    rm -rf /root/.cache/pip

# Copy app source code
COPY src/odoo_rag/__init__.py ./odoo_rag/
COPY src/odoo_rag/app.py ./odoo_rag/
COPY src/odoo_rag/llm.py ./odoo_rag/
COPY src/odoo_rag/retriever.py ./odoo_rag/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONPATH=/app

# Expose Gradio port
EXPOSE 7860

# Run Gradio app
CMD ["python", "-m", "odoo_rag.app"]
