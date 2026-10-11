"""Describe figures (charts, diagrams, photos, scanned pages) with a vision model so
they can be retrieved by text search."""
import re

from . import llm

PROMPT = """You are indexing a figure cropped from page {page} of a PDF so it can be found by search and used to answer questions later.

Text from the same page, for context (may include the figure's caption):
<page_text>
{page_text}
</page_text>

Describe the figure. Start with exactly one line `Type: <chart|diagram|photo|table|scanned-text|other>`, then:
- Title or caption, if any.
- For charts: chart type, axes with units, every series/legend entry, and all data values you can read, as a Markdown table. Then 1-3 sentences on the main trends or comparisons.
- For diagrams: the components and how they connect.
- For photos/illustrations: what is shown.
- Transcribe any other text in the image verbatim.

Only state what is visible; mark estimated values with "~"."""


def describe_figure(png_bytes: bytes, page: int, page_text: str) -> tuple[str, str]:
    """Return (figure_type, description)."""
    text = llm.generate([
        ("image", png_bytes),
        ("text", PROMPT.format(page=page, page_text=page_text[:2000] or "(none)")),
    ])
    if not text:
        return "figure", f"Figure on page {page} (no description available)."
    # Small models sometimes wrap the line in Markdown (`Type: chart` / **Type:** chart).
    match = re.match(r"[\s`*]*Type[\s`*]*:[\s`*]*([\w-]+)", text, re.IGNORECASE)
    fig_type = match.group(1).lower() if match else "figure"
    return fig_type, text
