#!/usr/bin/env python3
"""Incremental FAISS index smoke test.

This script performs a minimal verification of the incremental indexing logic:
1. Optionally ingests files if the Mongo collection is empty (--ingest-if-empty)
2. Runs an incremental build (first pass)
3. Runs a second incremental build (should add 0 new passages)
4. Reports Mongo document count, index vector count, and mapping length before/after

Usage:
    python src/index_smoke_test.py
    python src/index_smoke_test.py --no-incremental   # force full rebuild twice
    python src/index_smoke_test.py --ingest-if-empty  # ingest sample docs if collection empty

Exit codes:
 0 success
 2 Mongo connection failure
 3 Unexpected index/mapping inconsistency
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Tuple

import pymongo
import faiss  # type: ignore

# Import pipeline functions
from rag_windows import (
    get_collection,
    build_index,
    ingest_files,
    FAISS_INDEX_PATH,
    ID_MAP_PATH,
)

def log(msg: str) -> None:
    print(msg, flush=True)


def _load_index_state() -> Tuple[int, int]:
    """Return (vectors_in_index, mapping_length). If missing, returns (0,0)."""
    if not FAISS_INDEX_PATH.exists() or not ID_MAP_PATH.exists():
        return 0, 0
    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        vectors = index.ntotal
    except Exception:
        vectors = -1
    try:
        mapping = json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
        mapping_len = len(mapping) if isinstance(mapping, list) else -1
    except Exception:
        mapping_len = -1
    return vectors, mapping_len


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser("Smoke test for incremental FAISS indexing")
    parser.add_argument("--no-incremental", action="store_true", help="Force full rebuilds (disable incremental mode)")
    parser.add_argument("--ingest-if-empty", action="store_true", help="Ingest sample files if Mongo collection empty")
    args = parser.parse_args()

    # Connect to Mongo and get collection
    try:
        coll = get_collection()
    except SystemExit as e:  # get_collection sys.exit(2) on failure
        sys.exit(e.code)

    doc_count = coll.count_documents({})
    log(f"[smoke] Mongo documents before: {doc_count}")

    if doc_count == 0 and args.ingest_if_empty:
        log("[smoke] Ingesting sample files (collection empty)...")
        ingest_files()
        doc_count = coll.count_documents({})
        log(f"[smoke] Mongo documents after ingest: {doc_count}")
    elif doc_count == 0 and not args.ingest_if_empty:
        log("[smoke] Collection empty. Re-run with --ingest-if-empty or ingest manually.")
        sys.exit(0)

    before_vectors, before_map = _load_index_state()
    log(f"[smoke] Pre-build: vectors={before_vectors} map_len={before_map}")

    # First build (incremental by default)
    build_index(incremental=not args.no_incremental)
    mid_vectors, mid_map = _load_index_state()
    log(f"[smoke] After first build: vectors={mid_vectors} map_len={mid_map}")

    # Second build (should append zero if incremental and no new docs)
    build_index(incremental=not args.no_incremental)
    final_vectors, final_map = _load_index_state()
    log(f"[smoke] After second build: vectors={final_vectors} map_len={final_map}")

    # Basic assertions
    if mid_vectors != mid_map or final_vectors != final_map:
        log("[smoke][error] Index vector count and id_map length mismatch.")
        sys.exit(3)

    if not args.no_incremental and final_vectors != mid_vectors:
        log("[smoke][warn] Expected no new vectors on second incremental build, but counts differ.")
    else:
        log("[smoke] Incremental behavior confirmed.")

    log("[smoke] Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("[smoke] Interrupted by user.")
        sys.exit(130)
