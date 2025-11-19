# Use a stable slim image
FROM python:3.10-slim

WORKDIR /app

# Install minimal system deps for pillow/opencv functionality
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .

# Upgrade pip & install numpy first to avoid ABI mismatch (same trick as CI)
RUN pip install --upgrade pip setuptools wheel
RUN pip install "numpy==1.24.3"
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and saved model
COPY . .

# Expose Flask port
EXPOSE 5000

ENV FLASK_ENV=production
CMD ["python", "app.py"]
