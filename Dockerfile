# =============================================================================
# Wine Quality Prediction - Docker Image
# =============================================================================
# Build:  docker build -t wine-quality-predictor .
# Run:    docker run -v $(pwd):/data wine-quality-predictor
# =============================================================================

# Base image with PySpark pre-installed
FROM fokkodriesprong/docker-pyspark:latest

# Maintainer information
LABEL maintainer="Vedant Abrol <va398>"
LABEL description="CS643 Cloud Computing - Wine Quality Prediction with Apache Spark MLlib"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy application code
COPY docker/Predictions.py .
COPY TrainingDataset.csv .
COPY ValidationDataset.csv .

# Create data directory for volume mounting
RUN mkdir -p /data

# Copy default data files to /data (can be overridden by volume mount)
RUN cp TrainingDataset.csv ValidationDataset.csv /data/

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pyspark; print('OK')" || exit 1

# Default command - run predictions
CMD ["python", "Predictions.py"]
