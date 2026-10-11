# PDF RAG Chat

Chat with PDFs and get answers drawn from their **text, tables, charts and images**, with page citations.

## Features

- **Multimodal ingestion**: body text, tables (as Markdown plus an image snapshot), embedded images, and vector-drawn charts, which most generated PDFs use.
- **Figure understanding**: each chart or figure crop is described by a vision model (Groq by default, or Claude). The description covers chart type, axes, series, a data table and any text in the image. Scanned pages get transcribed the same way.
- **Hybrid retrieval**: local `bge-small` embeddings (fastembed/ONNX, no PyTorch) and BM25 keyword scores, combined with reciprocal rank fusion. No vector database needed.
- **Grounded answers**: answers stream in with inline `[p. N]` citations. Chart data comes from the figure descriptions, and figure images can also be attached at answer time (`RAG_MAX_IMAGES`).
- **Ingest-once cache**: processed PDFs are stored by content hash, so re-uploading a file costs nothing.
- **Two front ends**: a Streamlit web UI (`app.py`) and a terminal chat (`cli.py`).

## Architecture

```
PDF ──► ingest.py ──► chunks (text | table | figure) ──► index.py ──► data/<sha256>/
          │  PyMuPDF text / find_tables() / cluster_drawings()        chunks.json
          └─ vision.py: LLM describes each figure crop                vectors.npy
                                                                       images/*.png
question ──► index.search(): dense (fastembed)             + BM25 ──► RRF top-k
         ──► chat.py: excerpts (+ optional figure images) ──► LLM (streaming)        ──► answer [p. N]
```

1. **Ingest** ([rag_chat/ingest.py](rag_chat/ingest.py)), using PyMuPDF:
   - **Text** is split into overlapping chunks, one page at a time. Text inside tables and figures is left out.
   - **Tables** are found with `find_tables()` and stored as Markdown, plus a cropped snapshot of the table.
   - **Figures** cover both embedded raster images and vector-drawn charts. Nearby tick labels, axis titles and captions are included in the crop. Each crop is described by the vision model ([rag_chat/vision.py](rag_chat/vision.py)).
2. **Index** ([rag_chat/index.py](rag_chat/index.py)): local embeddings and BM25, fused with reciprocal rank fusion. BM25 catches exact numbers and names that embeddings tend to miss.
3. **Answer** ([rag_chat/chat.py](rag_chat/chat.py)): the top excerpts go to the model, optionally with the top figure images attached (`RAG_MAX_IMAGES`; off by default on Groq for speed) so chart values can be read from the image itself.
4. **LLM backend** ([rag_chat/llm.py](rag_chat/llm.py)): Groq (default, free tier) or Claude, chosen with `RAG_LLM_BACKEND`.
5. **Cache** ([rag_chat/pipeline.py](rag_chat/pipeline.py)): everything is written to `data/<sha256>/` and reused on later uploads.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env              # then put your GROQ_API_KEY in .env (free: https://console.groq.com/keys)
```

The embedding model (~70 MB) downloads on first run.

## Run

```bash
.venv/bin/streamlit run app.py                 # web UI: upload PDFs, chat, view sources
.venv/bin/python cli.py report.pdf             # terminal chat
.venv/bin/python cli.py report.pdf --inspect   # show what was extracted
```

If no API key is configured, the web UI shows setup instructions instead of failing.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (required for Groq) | Groq API key (free tier). Read from the environment, a local `.env`, or Streamlit secrets (`.streamlit/secrets.toml` / Streamlit Cloud). |
| `RAG_LLM_BACKEND` | `groq` | `groq` or `claude`. |
| `RAG_GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq vision model for figure descriptions and answers. |
| `RAG_MAX_IMAGES` | `0` on Groq, `4` on Claude | Chart/table images attached to each answer. |
| `RAG_GROQ_ANSWER_REASONING` | `default` | Reasoning for answers: `none`, `default`, `low`, `medium`, `high`. |
| `ANTHROPIC_API_KEY` | (required for Claude) | Only when `RAG_LLM_BACKEND=claude`. |
| `RAG_CLAUDE_MODEL` | `claude-opus-5-5` | Claude model, when using the Claude backend. |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (any fastembed-supported model). |
| `RAG_DATA_DIR` | `./data` | Where processed PDFs are cached. |
| `PORT` | `8501` | Port the Docker image listens on (set automatically by Railway). |

Chunk sizes, minimum figure size, `TOP_K` and the number of images attached per answer are set in [rag_chat/config.py](rag_chat/config.py).

## Deploy

### Railway (Docker)

The included [Dockerfile](Dockerfile) bakes the embedding model into the image, and starts Streamlit on `$PORT`.

1. Push this repo to GitHub and create a Railway project from it. Railway detects the Dockerfile automatically.
2. In **Variables**, set `GROQ_API_KEY` (and optionally `RAG_GROQ_MODEL`, etc.).
3. Under **Settings → Networking**, generate a public domain.
4. Optional: attach a volume mounted at `/app/data` so the PDF cache survives redeploys. Without one, the cache is lost on every deploy.

About 1 GB of RAM is enough: embeddings run on ONNX (no PyTorch).

Local test of the image:

```bash
docker build -t rag-chat .
docker run --rm -p 8501:8501 -e GROQ_API_KEY=gsk_... rag-chat
```

### Streamlit Community Cloud (alternative)

1. Create a new app from the GitHub repo with `app.py` as the entry point.
2. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
3. Deploy. The app reads the key from `st.secrets`. Streamlit Cloud installs from `requirements.txt`.

## Cost and rate limits

- Groq's free tier costs nothing, but it has per-minute and per-day request and token limits. Each image counts as about 2K input tokens, and the app retries automatically when it hits a limit. PDFs with many figures take longer to process on the free tier. Upgrading the Groq plan raises the limits.
- Ingest makes one vision call per figure. Text and tables need no LLM calls.
- Each answer sends about 6 excerpts. On Groq, answers are text-only by default (about 2 s). Figure descriptions written at ingest already contain the chart's data values. Set `RAG_MAX_IMAGES=1` to also attach the top chart image, which is more accurate on complex charts but takes about 30–45 s per answer on the free tier. Claude gets up to 4 images.
- With the Claude backend, requests use server-side refusal fallbacks (`fallbacks="default"`). If a safety classifier declines a request, the API retries it on Anthropic's recommended fallback model.

## Privacy

Uploaded PDFs are processed on the server running the app. Figure crops and retrieved excerpts are sent to the configured LLM API (Groq or Anthropic). Don't upload documents you aren't allowed to share with a third-party API.
