FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    RAG_DATA_DIR=/app/data

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so containers don't download it on every boot.
ARG RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${RAG_EMBED_MODEL}', cache_dir='/app/.cache/fastembed')"

COPY . .

EXPOSE 8501

# Railway injects $PORT; fall back to 8501 for local `docker run`.
CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
