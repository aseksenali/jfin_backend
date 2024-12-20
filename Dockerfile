FROM python:3.12-slim AS builder

# Set environment variables for Poetry
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.6.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

# Update PATH to include Poetry
ENV PATH="/opt/poetry/bin:$PATH"

# Install system dependencies and Poetry
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --only main

ENV MILVUS_URL="http://milvus-standalone:19530" \
    OLLAMA_URL="http://ollama:11434" \
    EMBEDDING_MODEL_NAME="intfloat/multilingual-e5-base" \
    MODEL_NAME="llama3.2" \
    BACKEND_URL="http://backend:5110" \
    REINDEX=true \
    SOURCE_DIRECTORY="/app/sources"
# Copy application code
COPY . .

# Expose backend port
EXPOSE 5110

ENTRYPOINT ["poetry", "run", "python", "-m", "jfin_gpt.api"]
