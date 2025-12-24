"""Download a suitable local GGUF LLM for generation.

This project is designed to run fully local. This script downloads a single
quantized GGUF file into the ./models folder.

Run:
  python download_llm_model.py

It will download:
  Qwen/Qwen2.5-3B-Instruct-GGUF - qwen2.5-3b-instruct-q4_k_m.gguf

Notes:
- Requires an internet connection *only while downloading*.
- After download, the API/CLI runs fully offline.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception:
        print("[error] Missing dependency: huggingface_hub", file=sys.stderr)
        print("        Install it via: pip install -U huggingface_hub", file=sys.stderr)
        return 2

    # Hugging Face increasingly serves large files via XET/CAS. On some networks this
    # can be flaky. Default to classic HTTP unless the user explicitly enables XET.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    filename = "qwen2.5-3b-instruct-q4_k_m.gguf"
    dest_path = models_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[ok] Model already present: {dest_path}")
        return 0

    print("[download] Downloading GGUF model...")
    print(f"           repo: {repo_id}")
    print(f"           file: {filename}")

    attempts = int(os.getenv("HF_DOWNLOAD_ATTEMPTS", "8"))
    base_sleep_s = float(os.getenv("HF_DOWNLOAD_BACKOFF_SECONDS", "3"))

    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(models_dir),
            )
            print(f"[ok] Downloaded -> {downloaded_path}")
            return 0
        except Exception as e:
            last_err = e
            print(f"[warn] Attempt {i}/{attempts} failed: {e}", file=sys.stderr)
            if i < attempts:
                sleep_s = base_sleep_s * (2 ** (i - 1))
                print(f"       Retrying in {sleep_s:.1f}s...", file=sys.stderr)
                time.sleep(sleep_s)

    print("[error] Download failed after multiple attempts.", file=sys.stderr)
    print("        Tips:", file=sys.stderr)
    print("        - Try a different network (CAS/CDN can be flaky)", file=sys.stderr)
    print("        - Ensure HF_HUB_DISABLE_XET=1 (this script sets it by default)", file=sys.stderr)
    print("        - You can increase retries: set HF_DOWNLOAD_ATTEMPTS=12", file=sys.stderr)
    if last_err is not None:
        print(f"        Last error: {last_err}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
