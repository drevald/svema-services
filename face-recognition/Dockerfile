# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for building dlib and face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Install dlib-bin first (pre-compiled binary) to avoid building from source
RUN pip install --no-cache-dir --prefix=/install dlib-bin==19.24.6

# Install face-recognition without dependencies to avoid building dlib from source
RUN pip install --no-cache-dir --prefix=/install --no-deps face-recognition==1.3.0 face-recognition-models

# Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install flask==3.0.0 numpy==1.24.3 pillow==10.1.0 scikit-learn==1.3.2 pytest==7.4.3 click

# Stage 2: Runtime image
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies (not build tools)
RUN apt-get update && apt-get install -y \

    libopenblas0 \
    liblapack3 \
    libgomp1 \
    libx11-6 \
    libpng16-16 \
    libjpeg62-turbo \
    libwebp7 \
    libgif7 \
    libtiff6 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Expose port
EXPOSE 5555

# Run the Flask app
CMD ["python", "app.py"]
