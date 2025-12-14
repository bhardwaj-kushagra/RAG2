#!/usr/bin/env python3
"""
Download a local LLM model in GGUF format for generation/agent.

This script downloads Meta-Llama-3.1-8B-Instruct (Q4_K_M quantized) which is:
- Strong instruction-following model for chat and simple agents
- Reasonable size for local CPU inference (quantized)

Usage:
    python download_llm_model.py
    python download_llm_model.py --force

After download, it will be stored in the models/ directory and used by default
via DEFAULT_MODEL_PATH in src/rag_windows.py.
"""
import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

LLM_MODEL = {
    "name": "meta-llama-3.1-8b-instruct-q4_k_m.gguf",
    "url": "https://huggingface.co/joshnader/Meta-Llama-3.1-8B-Instruct-Q4_K_M-GGUF/resolve/main/meta-llama-3.1-8b-instruct-q4_k_m.gguf",
    "size": "~5-6GB (approx)",
    "description": "Meta-Llama-3.1-8B-Instruct Q4_K_M for local chat/agent use",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)

    print(f"\u2713 Downloaded successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download local LLM model (Meta-Llama-3.1-8B-Instruct Q4_K_M)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    args = parser.parse_args()

    model_name = LLM_MODEL["name"]
    model_url = LLM_MODEL["url"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / model_name

    if output_path.exists() and not args.force:
        print(f"\u2713 LLM model already exists: {output_path}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024*1024):.2f} GB")
        print("  Use --force to re-download")
        print("\nTo use this model (already wired as default):")
        print(f"  python src/rag_windows.py --query \"What is FAISS?\" --model-path \"{output_path}\"")
        return

    print("=" * 60)
    print("Downloading LLM Model: Meta-Llama-3.1-8B-Instruct-Q4_K_M")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Size: {LLM_MODEL['size']}")
    print(f"Description: {LLM_MODEL['description']}")
    print("=" * 60)
    print()

    try:
        download_file(model_url, output_path)

        print()
        print("=" * 60)
        print("SUCCESS! LLM model ready to use")
        print("=" * 60)
        print(f"Model location: {output_path}")
        print()
        print("Next steps:")
        print(f"  1. Ask a question: python src/rag_windows.py --query \"What is FAISS?\" --model-path \"{output_path}\"")
        print("  2. Use the agent:  python src/rag_windows.py --agent \"Run a quick evaluation of our RAG quality.\" --model-path \"{output_path}\"")
    except Exception as e:
        print(f"\n\u2717 Download failed: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Verify the URL is accessible")
        sys.exit(1)


if __name__ == "__main__":
    main()
