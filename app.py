"""Streamlit UI: upload PDFs, then chat with answers grounded in their text, tables and figures.

Run with:  streamlit run app.py
"""
import os
from collections import Counter
from pathlib import Path

import streamlit as st

from rag_chat import chat, config, llm
from rag_chat.index import Index, embedder
from rag_chat.ingest import Chunk
from rag_chat.pipeline import load_or_build

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")

KEY_URL = "https://console.groq.com/keys" if config.BACKEND == "groq" else "https://console.anthropic.com/"
KEY_HELP = (
    f"Set `{config.API_KEY_ENV}` as an environment variable (or in a local `.env` file, see `.env.example`), "
    "or add it to `.streamlit/secrets.toml` / your Streamlit Cloud app secrets, then restart the app. "
    f"Get a key at {KEY_URL}."
)


def ensure_api_key() -> bool:
    """Make the backend's API key available to its SDK from the environment or st.secrets."""
    if os.getenv(config.API_KEY_ENV) or (config.BACKEND == "claude" and os.getenv("ANTHROPIC_AUTH_TOKEN")):
        return True
    try:
        key = st.secrets.get(config.API_KEY_ENV)
    except Exception:  # no secrets.toml present
        key = None
    if key:
        os.environ[config.API_KEY_ENV] = str(key)
        return True
    return False


@st.cache_resource(show_spinner="Loading embedding model…")
def warm_embedder():
    return embedder()


def render_source(n: int, chunk: Chunk, score: float) -> None:
    label = chunk.subtype or chunk.kind
    with st.expander(f"[{n}] {chunk.doc} · p. {chunk.page} · {label}  (score {score:.3f})"):
        if chunk.image and Path(chunk.image).exists():
            st.image(chunk.image)
        if chunk.kind == "figure":
            st.caption("Description generated at ingest time")
        st.markdown(chunk.content)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    top_k = st.slider("Excerpts per answer", 3, 12, config.TOP_K)
    st.caption(f"Model: `{config.LLM_NAME}` · Embeddings: `{config.EMBED_MODEL}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexes", {})

if not ensure_api_key():
    st.title("📄 Chat with your PDF")
    st.warning(f"**No `{config.API_KEY_ENV}` configured.** {KEY_HELP}")
    st.stop()

warm_embedder()

if uploads:
    for up in uploads:
        if up.name in st.session_state.indexes:
            continue
        with st.status(f"Processing {up.name}…", expanded=True) as status:
            bar = st.progress(0.0)
            try:
                index = load_or_build(up.getvalue(), up.name, lambda msg, f: bar.progress(min(f, 1.0), text=msg))
            except Exception as e:
                if (message := llm.friendly_error(e)) is None:
                    raise
                status.update(label=f"Failed to process {up.name}", state="error")
                st.error(message)
                st.stop()
            st.session_state.indexes[up.name] = index
            status.update(label=f"Indexed {up.name}", state="complete", expanded=False)

# Forget documents that were removed from the uploader.
current = {up.name for up in uploads or []}
for name in list(st.session_state.indexes):
    if name not in current:
        del st.session_state.indexes[name]

with st.sidebar:
    for name, ix in st.session_state.indexes.items():
        counts = Counter(c.kind for c in ix.chunks)
        pages = max((c.page for c in ix.chunks), default=0)
        st.markdown(
            f"**{name}**  \n{pages} pages · {counts['text']} text chunks · "
            f"{counts['table']} tables · {counts['figure']} figures"
        )

# ---------------------------------------------------------------- chat
st.title("📄 Chat with your PDF")

if not st.session_state.indexes:
    st.info("Upload a PDF in the sidebar to start. Text, tables, charts and images are all indexed.")
    st.stop()

index = Index.merge(list(st.session_state.indexes.values()))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for n, (chunk, score) in enumerate(msg.get("sources", []), start=1):
            render_source(n, chunk, score)

if question := st.chat_input("Ask about the document…"):
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("assistant"):
        try:
            hits, stream = chat.answer(question, index, history, k=top_k)
            reply = st.write_stream(stream)
        except Exception as e:
            if (message := llm.friendly_error(e)) is None:
                raise
            st.error(message)
            st.stop()
        for n, (chunk, score) in enumerate(hits, start=1):
            render_source(n, chunk, score)

    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply, "sources": hits},
    ]
