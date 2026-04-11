# Use Python 3.10 slim as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the Flask application
COPY backend/ /app/backend/

# Copy the fine-tuned GLiNER model so it's baked into the image
COPY model/ /app/model/

# Set environment variables for production (optional, good practice)
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Start Flask via Gunicorn (a production WSGI server) or fallback to basic python
# We'll install gunicorn directly here just in case it's not in requirements.txt
RUN pip install --no-cache-dir gunicorn
WORKDIR /app/backend
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
