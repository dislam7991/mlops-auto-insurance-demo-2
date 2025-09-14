# FastAPI Uvicorn serving image
# Build with: docker build -t autoins-svc -f Dockerfile .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    WORKERS=1 \
    MODEL_PATH=/app/models/model.pkl

WORKDIR /app

# System deps (kept minimal; add more if your wheels need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install only serving dependencies to keep image small
COPY requirements-serving.txt ./
RUN pip install --upgrade pip && pip install -r requirements-serving.txt

# Copy app source
COPY autoinsurance ./autoinsurance

# Create models dir (mount or copy your model here at runtime)
RUN mkdir -p /app/models

EXPOSE 8000

# Default command uses the module's __main__ to read HOST/PORT/RELOAD envs
CMD ["python", "-m", "autoinsurance.serving.app"]
