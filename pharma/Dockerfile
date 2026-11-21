# ==========================================================
# 🧠 Pharma Weekly Forecast API - Dockerfile (Python 3.11)
# ==========================================================

# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask default port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
