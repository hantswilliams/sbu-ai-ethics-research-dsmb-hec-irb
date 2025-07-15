# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=webapp/app.py \
    FLASK_ENV=production \
    PORT=5005

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY webapp/requirements.txt requirements.webapp.txt
COPY code/requirements.txt requirements.code.txt

# Create a combined requirements file without duplicates
RUN cat requirements.webapp.txt requirements.code.txt | sort -u > requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared module and application code
COPY shared/ /app/shared/
COPY webapp/ /app/webapp/

# Make port 5005 available to the world outside this container
EXPOSE 5005

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run the application
CMD ["python", "webapp/app.py"]
