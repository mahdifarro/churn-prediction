
# Optimized Dockerfile for smaller image size
FROM python:3.11-slim

# Install system dependencies for lightgbm and xgboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# Copy only necessary source files
COPY ./src ./src
COPY ./api ./api

# Set environment variables for Python for cleaner logging and no .pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the port the app will run on
EXPOSE 8000

# The command to run when the container starts
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]