"""LLM backends. Callers pass content as a list of parts, ("text", str) or ("image", png
bytes), in reading order; each backend converts that to its own message format.

- groq:   Groq API (free tier), needs GROQ_API_KEY (default)
- claude: Anthropic API, needs ANTHROPIC_API_KEY
"""
import base64
import io
from functools import lru_cache
from typing import Iterator

from PIL import Image

from . import config

Part = tuple[str, "str | bytes"]


# ---------------------------------------------------------------- public API

def generate(parts: list[Part], *, max_tokens: int = 8000) -> str:
    """One-shot completion (used for figure descriptions at ingest)."""
    if config.BACKEND == "groq":
        response = _groq().chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": _groq_content(parts)}],
            reasoning_effort=config.GROQ_CAPTION_REASONING,
            reasoning_format="hidden",
            max_completion_tokens=max_tokens,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    message = _claude().beta.messages.create(
        **_claude_kwargs(config.CAPTION_EFFORT), max_tokens=max_tokens,
        messages=[{"role": "user", "content": _claude_content(parts)}],
    )
    if message.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in message.content if b.type == "text").strip()


def stream(system: str, history: list[dict], parts: list[Part]) -> Iterator[str]:
    """Stream an answer. `history` holds prior turns as {"role", "content": str}."""
    if config.BACKEND == "groq":
        finish = None
        for chunk in _groq().chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": system}, *history,
                      {"role": "user", "content": _groq_content(parts)}],
            reasoning_effort=config.GROQ_ANSWER_REASONING,
            reasoning_format="hidden",
            max_completion_tokens=8192,
            temperature=0.2,
            stream=True,
        ):
            if not chunk.choices:
                continue
            yield chunk.choices[0].delta.content or ""
            finish = chunk.choices[0].finish_reason or finish
        if finish == "length":
            yield "\n\n_(Answer truncated: output limit reached.)_"
        return

    messages = [*history, {"role": "user", "content": _claude_content(parts)}]
    with _claude().beta.messages.stream(
        **_claude_kwargs(config.ANSWER_EFFORT), max_tokens=64000, system=system, messages=messages,
    ) as s:
        yield from s.text_stream
        final = s.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n_The model declined to answer this request._"
    elif final.stop_reason == "max_tokens":
        yield "\n\n_(Answer truncated: output limit reached.)_"


def friendly_error(exc: Exception) -> "str | None":
    """A user-facing explanation for known setup/API failures, else None."""
    key = config.API_KEY_ENV
    if config.BACKEND == "groq":
        import groq
        if isinstance(exc, groq.AuthenticationError) or (
            isinstance(exc, groq.GroqError) and "api_key" in str(exc)
        ):
            return f"Groq authentication failed. Set `{key}` (free key at https://console.groq.com/keys)."
        if isinstance(exc, groq.APIStatusError) and exc.status_code == 413:
            return ("This question needs more tokens than Groq's free tier allows per minute. "
                    "Lower 'Excerpts per answer' or set RAG_MAX_IMAGES=0, or upgrade the Groq plan.")
        if isinstance(exc, groq.RateLimitError):
            return "Groq free-tier rate limit reached. Wait a minute and try again."
        if isinstance(exc, groq.APIError):
            return f"Groq API error: {exc}"
        return None

    import anthropic
    # Bad key -> AuthenticationError; no credentials at all -> TypeError from the SDK.
    if isinstance(exc, anthropic.AuthenticationError) or (
        isinstance(exc, TypeError) and "authentication" in str(exc).lower()
    ):
        return f"Claude API authentication failed. Set `{key}`."
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by the Claude API. Wait a moment and try again."
    if isinstance(exc, anthropic.APIError):
        return f"Claude API error: {exc}"
    return None


# ---------------------------------------------------------------- groq

@lru_cache(maxsize=1)
def _groq():
    import groq
    # Reads GROQ_API_KEY. Extra retries ride out free-tier per-minute limits.
    return groq.Groq(max_retries=4)


def _jpeg_data_url(png: bytes) -> str:
    # JPEG keeps multi-image requests well under Groq's request-size limit.
    img = Image.open(io.BytesIO(png)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _groq_content(parts: list[Part]) -> list[dict]:
    return [
        {"type": "image_url", "image_url": {"url": _jpeg_data_url(value)}} if kind == "image"
        else {"type": "text", "text": value}
        for kind, value in parts
    ]


# ---------------------------------------------------------------- claude

# Server-side refusal fallback: if a safety classifier declines a request, the API
# re-runs it on Anthropic's recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@lru_cache(maxsize=1)
def _claude():
    import anthropic
    # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
    return anthropic.Anthropic()


def _claude_kwargs(effort: str) -> dict:
    return {
        "model": config.CLAUDE_MODEL,
        "output_config": {"effort": effort},
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }


def _claude_content(parts: list[Part]) -> list[dict]:
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.standard_b64encode(value).decode("utf-8")}}
        if kind == "image" else {"type": "text", "text": value}
        for kind, value in parts
    ]
