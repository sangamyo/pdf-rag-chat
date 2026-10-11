FROM python:3.13-slim

# XSRF protection is off because Hugging Face Spaces serves the app inside an iframe,
# where Streamlit's XSRF check rejects file uploads.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    RAG_DATA_DIR=/app/data \
    HOME=/home/user

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so containers don't download it on every boot.
ARG RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${RAG_EMBED_MODEL}', cache_dir='/app/.cache/fastembed')"

COPY . .
RUN mkdir -p /app/data && chown -R user:user /app
USER user

EXPOSE 8501

# Railway injects $PORT; Hugging Face Spaces uses app_port 8501; local `docker run` defaults to 8501.
CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
