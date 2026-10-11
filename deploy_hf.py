"""Deploy this app to a Hugging Face Docker Space.

    hf auth login                      # once, with a write token
    python deploy_hf.py                # -> https://huggingface.co/spaces/<you>/pdf-rag-chat

Uploads the git-tracked files, adds the Space config to the README, and copies
GROQ_API_KEY (from the environment or .env) into the Space's secrets.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent

SPACE_CONFIG = """---
title: PDF RAG Chat
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Chat with PDF text, tables, charts and images
---

"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pdf-rag-chat", help="Space name")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{args.name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        for name in files:
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, staging / name)
        readme = staging / "README.md"
        readme.write_text(SPACE_CONFIG + readme.read_text())
        api.upload_folder(folder_path=staging, repo_id=repo_id, repo_type="space",
                          commit_message="Deploy from local repo")

    for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        if os.getenv(key):
            api.add_space_secret(repo_id, key, os.environ[key])
            print(f"Set Space secret {key}")

    print(f"Space:  https://huggingface.co/spaces/{repo_id}")
    print(f"Direct: https://{repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
