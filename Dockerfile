# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download pt_core_news_lg
RUN python -m spacy download en_core_web_lg

# Create non-root user
RUN useradd -m -u 1000 lexa
USER lexa

# Expose port
EXPOSE 8000

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
