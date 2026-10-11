import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))

# "groq" (default, free tier) or "claude". Each needs its own API key.
BACKEND = os.getenv("RAG_LLM_BACKEND", "groq").lower()
API_KEY_ENV = "GROQ_API_KEY" if BACKEND == "groq" else "ANTHROPIC_API_KEY"

# Groq vision-language model (also used for answers, which include page images).
GROQ_MODEL = os.getenv("RAG_GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_CAPTION_REASONING = os.getenv("RAG_GROQ_CAPTION_REASONING", "none")
GROQ_ANSWER_REASONING = os.getenv("RAG_GROQ_ANSWER_REASONING", "default")

# Claude model used for figure descriptions and for answering.
CLAUDE_MODEL = os.getenv("RAG_CLAUDE_MODEL", "claude-opus-5-5")
LLM_NAME = GROQ_MODEL if BACKEND == "groq" else CLAUDE_MODEL
# Effort for one-off figure descriptions at ingest time (retrieval captions only;
# the image itself is re-read at answer time) and for answering questions.
CAPTION_EFFORT = os.getenv("RAG_CAPTION_EFFORT", "low")
ANSWER_EFFORT = os.getenv("RAG_ANSWER_EFFORT", "medium")

# Local embedding model (fastembed / ONNX).
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH")  # None -> fastembed's default
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking / extraction.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
MIN_FIGURE_PT = 72          # ignore figures smaller than 1 inch on either side
MAX_IMAGE_EDGE_PX = 1568    # long-edge cap for images sent to the model
# Groq's free tier has low per-minute limits, so describe figures two at a time.
VISION_WORKERS = 2 if BACKEND == "groq" else 4

# Retrieval.
TOP_K = 6
# Images attached to each answer (charts first; tables are already text). On Groq an
# attached image slowed answers from ~2s to ~30-45s and bills ~2K tokens against the free
# tier's ~7K/minute, while the ingest-time figure descriptions already carry the data values.
MAX_IMAGES_PER_ANSWER = int(os.getenv("RAG_MAX_IMAGES", "0" if BACKEND == "groq" else "4"))
# Prior chat messages replayed to the model (retrieved excerpts are never replayed).
HISTORY_MESSAGES = 4
