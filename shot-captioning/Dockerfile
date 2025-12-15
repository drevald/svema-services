# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5556

# Run the Flask app
CMD ["python", "app.py"]
