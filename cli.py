"""Terminal chat over one or more PDFs.

    python cli.py report.pdf [more.pdf ...]
    python cli.py report.pdf --inspect      # print extracted chunks, no chat
"""
import argparse
from pathlib import Path

from rag_chat import chat
from rag_chat.index import Index
from rag_chat.pipeline import load_or_build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--inspect", action="store_true", help="list extracted chunks and exit")
    args = parser.parse_args()

    indexes = []
    for pdf in args.pdfs:
        print(f"Indexing {pdf.name}…")
        indexes.append(load_or_build(pdf.read_bytes(), pdf.name, lambda msg, f: print(f"  {msg}")))
    index = Index.merge(indexes)

    if args.inspect:
        for c in index.chunks:
            print(f"\n=== {c.doc} p.{c.page} {c.kind}{'/' + c.subtype if c.subtype else ''} [{c.id}] ===")
            print(c.content[:600])
        return

    history: list[dict] = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        hits, stream = chat.answer(question, index, history)
        print("Claude: ", end="", flush=True)
        reply = "".join(print(t, end="", flush=True) or t for t in stream)
        print("\n  sources: " + ", ".join(f"p.{c.page} {c.kind}" for c, _ in hits))
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
