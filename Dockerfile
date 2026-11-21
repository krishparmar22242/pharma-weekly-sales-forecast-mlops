# ================================
# 🧠 Pharma Forecast MLOps API
# ================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI default port
EXPOSE 8000

# Default command: serve API
CMD ["python", "serve.py"]
