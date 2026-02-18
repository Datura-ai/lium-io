FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml pdm.lock* ./

# Install PDM and dependencies
RUN pip install --no-cache-dir pdm && \
    pdm install --prod --no-lock --no-editable

# Copy source code
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run watchtower
CMD ["pdm", "run", "src/watchtower.py"]
