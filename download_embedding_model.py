#!/usr/bin/env python3
"""
Download a local embedding model in GGUF format for text embeddings.

This script downloads nomic-embed-text-v1.5 (Q4_K_M quantized, ~275MB) which is:
- Optimized for semantic search and retrieval
- Small footprint with good quality
- Compatible with llama.cpp embedding mode

Alternative models you can use instead:
- all-MiniLM-L6-v2.Q4_K_M.gguf (~23MB) - smaller, faster
- bge-small-en-v1.5.Q4_K_M.gguf (~120MB) - good balance

Usage:
    python download_embedding_model.py
    python download_embedding_model.py --model all-minilm  # for smaller model
"""
import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

# Available embedding models
MODELS = {
    "nomic": {
        "name": "nomic-embed-text-v1.5.Q4_K_M.gguf",
        "url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf",
        "size": "275MB",
        "dims": 768,
        "description": "High quality embedding model optimized for search"
    },
    "all-minilm": {
        "name": "all-MiniLM-L6-v2.Q4_K_M.gguf",
        "url": "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q4_K_M.gguf",
        "size": "23MB",
        "dims": 384,
        "description": "Small, fast embedding model (same as SentenceTransformers default)"
    },
    "bge-small": {
        "name": "bge-small-en-v1.5.Q4_K_M.gguf",
        "url": "https://huggingface.co/BAAI/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q4_k_m.gguf",
        "size": "120MB",
        "dims": 384,
        "description": "Balanced quality/size embedding model"
    }
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
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"✓ Downloaded successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download local embedding model for RAG")
    parser.add_argument(
        "--model",
        type=str,
        default="all-minilm",
        choices=list(MODELS.keys()),
        help="Which embedding model to download (default: all-minilm, smallest)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    
    args = parser.parse_args()
    
    # Get model info
    model_info = MODELS[args.model]
    model_name = model_info["name"]
    model_url = model_info["url"]
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = MODELS_DIR / model_name
    
    # Check if already exists
    if output_path.exists() and not args.force:
        print(f"✓ Model already exists: {output_path}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"  Use --force to re-download")
        print(f"\nTo use this model:")
        print(f"  python src/rag_windows.py --build-index --embed-model-path \"{output_path}\"")
        return
    
    print("=" * 60)
    print(f"Downloading Embedding Model: {args.model}")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Size: {model_info['size']}")
    print(f"Dimensions: {model_info['dims']}")
    print(f"Description: {model_info['description']}")
    print("=" * 60)
    print()
    
    try:
        download_file(model_url, output_path)
        
        print()
        print("=" * 60)
        print("SUCCESS! Embedding model ready to use")
        print("=" * 60)
        print(f"Model location: {output_path}")
        print()
        print("Next steps:")
        print(f"  1. Rebuild index: python src/rag_windows.py --build-index --no-incremental --embed-model-path \"{output_path}\"")
        print(f"  2. Query: python src/rag_windows.py --query \"What is FAISS?\" --embed-model-path \"{output_path}\"")
        print()
        print("Or update your code to use this as default:")
        print(f"  EMBED_MODEL_PATH = PROJECT_ROOT / \"models\" / \"{model_name}\"")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Verify the URL is accessible")
        print("  - Try a different model with --model flag")
        sys.exit(1)


if __name__ == "__main__":
    main()
