# Stage 1: Download model
FROM python:3.11-slim AS model-downloader

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time to speed up container startup
RUN python -c "from transformers import Blip2Processor, Blip2ForConditionalGeneration; \
    import torch; \
    model_name='Salesforce/blip2-flan-t5-xl'; \
    Blip2Processor.from_pretrained(model_name); \
    Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)"

# Stage 2: Final image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model cache from model-downloader stage
COPY --from=model-downloader /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
