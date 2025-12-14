from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

import rag_windows


@dataclass
class ToolResult:
    tool: str
    args: Dict[str, Any]
    result: Any


# --- Simple tool wrappers ---


def _get_mongo_client() -> MongoClient:
    return MongoClient(rag_windows.MONGO_URI, serverSelectionTimeoutMS=3000)


def search_docs(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Use existing FAISS + Mongo retrieval and return lightweight dicts."""
    passages = rag_windows.retrieve(query, top_k=max(1, k))
    return [
        {"mongo_id": p.mongo_id, "source_id": p.source_id, "text": p.text}
        for p in passages
    ]


def get_raw_from_db(filter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch raw documents from rag_db.passages with a Mongo-style filter."""
    client = _get_mongo_client()
    db = client[rag_windows.DB_NAME]
    coll = db[rag_windows.COLLECTION_NAME]
    docs: List[Dict[str, Any]] = []
    for d in coll.find(filter).limit(20):  # hard cap for safety
        d["_id"] = str(d.get("_id"))
        docs.append(d)
    return docs


def write_note_to_db(text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Insert a small note into rag_db.notes for tracking / planning."""
    client = _get_mongo_client()
    db = client[rag_windows.DB_NAME]
    coll = db["notes"]
    payload: Dict[str, Any] = {"text": text}
    if meta:
        payload.update(meta)
    res = coll.insert_one(payload)
    return {"inserted_id": str(res.inserted_id)}


TOOLS = {
    "search_docs": search_docs,
    "get_raw_from_db": get_raw_from_db,
    "write_note_to_db": write_note_to_db,
}


def _classify_goal(goal: str) -> str:
    """Very simple rule-based classifier to pick an agent mode.

    Returns one of: "evaluation", "note", "inspect_db", "search_summarize".
    """
    g = goal.lower()

    if any(kw in g for kw in ["ragas", "evaluation", "evaluate quality", "rag quality", "run evaluation"]):
        return "evaluation"

    if any(kw in g for kw in ["note", "todo", "task", "remind", "remember", "create a note", "add a note"]):
        return "note"

    if any(kw in g for kw in ["inspect db", "inspect database", "raw db", "raw documents", "what is in rag_db.passages", "what data is stored in rag_db.passages", "show passages"]):
        return "inspect_db"

    return "search_summarize"


def _build_summary_prompt(goal: str, passages: List[Dict[str, Any]]) -> str:
    """Build a prompt that shows retrieved passages and asks for a concise summary."""
    context_blocks: List[str] = []
    for p in passages:
        sid = p.get("source_id", "?")
        text = p.get("text", "")
        preview = text if len(text) <= 600 else text[:600] + "..."
        context_blocks.append(f"[{sid}]\n{preview}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "(no relevant passages)"

    return (
        "You are a helpful assistant working over a local RAG system.\n"
        "You are given the user's goal and some passages retrieved from the knowledge base.\n"
        "Using ONLY this information, answer the goal concisely in 2-4 sentences.\n"
        "If there is not enough information, say so explicitly.\n\n"
        f"USER GOAL:\n{goal}\n\n"
        f"RETRIEVED PASSAGES:\n{context_text}\n\n"
        "ANSWER:"
    )


def _build_db_inspect_prompt(goal: str, docs: List[Dict[str, Any]]) -> str:
    """Prompt to summarize raw DB documents for a goal like "inspect db"."""
    blocks: List[str] = []
    for d in docs:
        # Shallow copy with limited fields, convert non-serializable types
        clipped_raw = {k: v for k, v in d.items() if k in {"_id", "source", "source_id", "text", "created_at"}}
        clipped: Dict[str, Any] = {}
        for k, v in clipped_raw.items():
            if hasattr(v, "isoformat"):
                clipped[k] = v.isoformat()
            else:
                clipped[k] = v

        text = str(clipped.get("text", ""))
        if len(text) > 300:
            clipped["text"] = text[:300] + "..."
        preview = json.dumps(clipped, ensure_ascii=False)
        blocks.append(preview)

    context = "\n\n".join(blocks) if blocks else "(no documents)"

    return (
        "You are inspecting raw documents from the MongoDB collection rag_db.passages.\n"
        "Summarize what kind of data is stored there, what the typical fields mean, and how this data is used in a RAG pipeline.\n"
        "Be concise (3-5 sentences).\n\n"
        f"USER GOAL:\n{goal}\n\n"
        f"SAMPLED DOCUMENTS:\n{context}\n\n"
        "ANSWER:"
    )


def run_agent(goal: str, model_path: Optional[Path] = None, k: int = 5) -> str:
    """Rule-based multi-tool agent.

    Modes:
      - evaluation: run simplified RAG evaluation and summarize metrics.
      - note: store a note in rag_db.notes.
      - inspect_db: sample raw docs from rag_db.passages and summarize them.
      - search_summarize: search_docs() then summarize retrieved passages.
    """
    mp = model_path or rag_windows.DEFAULT_MODEL_PATH
    mode = _classify_goal(goal)
    rag_windows.log(f"[agent] Classified goal as mode='{mode}'")

    # 1) Run simplified evaluation
    if mode == "evaluation":
        try:
            import evaluation as eval_module

            rag_windows.log("[agent] Running simplified evaluation via evaluation.run_evaluation_simple()")
            results = eval_module.run_evaluation_simple(embed_model_path=str(eval_module.DEFAULT_EMBED_MODEL_PATH))
            return (
                "I ran the simplified RAG evaluation. Here are the aggregate metrics: "
                f"Faithfulness={results.get('faithfulness'):.3f}, "
                f"Answer Relevancy={results.get('answer_relevancy'):.3f}, "
                f"Context Precision={results.get('context_precision'):.3f}, "
                f"Context Recall={results.get('context_recall'):.3f} "
                f"over {results.get('num_samples')} samples."
            )
        except Exception as e:
            rag_windows.log(f"[agent] Evaluation mode failed: {e}")
            return "The agent tried to run the evaluation but encountered an error. Check logs for details."

    # 2) Create a note in rag_db.notes
    if mode == "note":
        try:
            meta = {"created_by": "agent", "kind": "note"}
            rag_windows.log("[agent] Writing note to rag_db.notes")
            res = write_note_to_db(text=goal, meta=meta)
            note_id = res.get("inserted_id")
            return f"I stored a note in rag_db.notes with id {note_id} capturing this goal: '{goal}'."
        except Exception as e:
            rag_windows.log(f"[agent] Note mode failed: {e}")
            return "The agent tried to write a note to the database but failed."

    # 3) Inspect raw documents from rag_db.passages
    if mode == "inspect_db":
        try:
            rag_windows.log("[agent] Sampling raw documents from rag_db.passages")
            docs = get_raw_from_db({})
            if not docs:
                return "I could not find any documents in rag_db.passages to inspect."

            llm = rag_windows._load_llm(mp)
            prompt = _build_db_inspect_prompt(goal, docs)
            out = llm.create_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=0.2,
                top_p=0.9,
            )
            return out["choices"][0]["text"].strip()
        except Exception as e:
            rag_windows.log(f"[agent] Inspect_db mode failed: {e}")
            return "The agent tried to inspect the database but failed."

    # 4) Default: search + summarize
    rag_windows.log(f"[agent] Using search_docs for goal: {goal}")
    passages = search_docs(goal, k=max(1, k))

    if not passages:
        return "I could not find any relevant passages in the knowledge base for this goal."

    llm = rag_windows._load_llm(mp)
    prompt = _build_summary_prompt(goal, passages)

    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
        )
        return out["choices"][0]["text"].strip()
    except Exception as e:
        rag_windows.log(f"[agent] LLM summarization failed: {e}")
        return "The agent failed while trying to summarize the retrieved information."