from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from bson import ObjectId

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


def tag_top_passages(query: str, tag: str, k: int = 5) -> Dict[str, Any]:
    """Tag the top-k passages for a query with a label in MongoDB."""
    client = _get_mongo_client()
    db = client[rag_windows.DB_NAME]
    coll = db[rag_windows.COLLECTION_NAME]

    hits = search_docs(query, k=max(1, k))
    ids = [ObjectId(p["mongo_id"]) for p in hits if p.get("mongo_id")]
    if not ids:
        return {"matched": 0, "modified": 0, "tag": tag, "query": query}

    res = coll.update_many({"_id": {"$in": ids}}, {"$addToSet": {"tags": tag}})
    return {
        "matched": int(res.matched_count),
        "modified": int(res.modified_count),
        "tag": tag,
        "query": query,
    }


def rebuild_full_index(batch_size: int = 64) -> Dict[str, Any]:
    """Force a full FAISS index rebuild using rag_windows.build_index."""
    rag_windows.log("[agent] Rebuilding FAISS index (full, non-incremental)...")
    rag_windows.build_index(batch_size=batch_size, incremental=False)
    return {"status": "ok", "batch_size": batch_size}


def add_eval_sample_from_goal(goal: str, k: int = 3) -> Dict[str, Any]:
    """Create a simple evaluation sample document in Mongo from a goal."""
    client = _get_mongo_client()
    db = client[rag_windows.DB_NAME]
    coll = db["eval_samples"]

    passages = search_docs(goal, k=max(1, k))
    contexts = [p.get("text", "") for p in passages]

    doc: Dict[str, Any] = {
        "question": goal,
        "contexts": contexts,
        "created_at": datetime.utcnow(),
        "created_by": "agent",
    }
    res = coll.insert_one(doc)
    return {"inserted_id": str(res.inserted_id), "num_contexts": len(contexts)}


TOOLS = {
    "search_docs": search_docs,
    "get_raw_from_db": get_raw_from_db,
    "write_note_to_db": write_note_to_db,
    "tag_top_passages": tag_top_passages,
    "rebuild_full_index": rebuild_full_index,
    "add_eval_sample_from_goal": add_eval_sample_from_goal,
}


def _classify_goal(goal: str) -> str:
    """Very simple rule-based classifier to pick an agent mode."""
    g = goal.lower()

    if any(kw in g for kw in ["ragas", "full ragas", "ragas evaluation"]):
        return "full_ragas"

    if any(kw in g for kw in ["evaluation", "evaluate quality", "rag quality", "run evaluation", "quick evaluation"]):
        return "evaluation"

    if any(kw in g for kw in ["note", "todo", "task", "remind", "remember", "create a note", "add a note"]):
        return "note"

    if any(kw in g for kw in [
        "inspect db",
        "inspect database",
        "raw db",
        "raw documents",
        "what is in rag_db.passages",
        "what data is stored in rag_db.passages",
        "show passages",
    ]):
        return "inspect_db"

    if any(kw in g for kw in ["rebuild index", "rebuild faiss", "re-index all", "reindex all"]):
        return "rebuild_index"

    if any(kw in g for kw in ["tag passages", "tag docs", "label passages", "label documents"]):
        return "tag_passages"

    if any(kw in g for kw in ["evaluation set", "eval set", "add eval sample", "save for evaluation"]):
        return "manage_eval"

    if any(kw in g for kw in ["multi-step", "multi step", "tool plan", "plan steps", "use tools"]):
        return "planned"

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
    """Prompt to summarize raw DB documents for a goal like 'inspect db'."""
    blocks: List[str] = []
    for d in docs:
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
      - full_ragas: run full RAGAS evaluation via ragas_evaluator when requested.
      - evaluation: run simplified RAG evaluation and summarize metrics.
      - note: store a note in rag_db.notes.
      - inspect_db: sample raw docs from rag_db.passages and summarize them.
      - rebuild_index: force a full FAISS index rebuild.
      - tag_passages: tag the top-k passages for a query.
    - manage_eval: create simple evaluation samples in Mongo.
    - planned: simple multi-step, LLM-planned tool use.
    - search_summarize: search_docs() then summarize retrieved passages.
    """
    mp = model_path or rag_windows.DEFAULT_MODEL_PATH
    mode = _classify_goal(goal)
    rag_windows.log(f"[agent] Classified goal as mode='{mode}'")

    # 1) Run full RAGAS evaluation when explicitly requested
    if mode == "full_ragas":
        try:
            import ragas_evaluator as ragas_mod

            rag_windows.log("[agent] Running full RAGAS evaluation via ragas_evaluator.run_ragas_evaluation()")
            samples = ragas_mod.load_test_data(None)
            results = ragas_mod.run_ragas_evaluation(samples)

            scores = results.get("scores") or {}
            num_samples = results.get("num_samples", len(samples))
            used = results.get("metrics_used") or []

            return (
                "I ran the full RAGAS evaluation over the test set. "
                f"Metrics used: {', '.join(used) or 'N/A'}. "
                f"Faithfulness={scores.get('faithfulness', 'n/a')}, "
                f"Answer Relevancy={scores.get('answer_relevancy', 'n/a')}, "
                f"Context Precision={scores.get('context_precision', 'n/a')}, "
                f"Context Recall={scores.get('context_recall', 'n/a')} "
                f"over {num_samples} samples. "
                "If OPENAI_API_KEY was not set, these scores come from the mock evaluator."
            )
        except Exception as e:
            rag_windows.log(f"[agent] Full RAGAS evaluation failed: {e}")
            return "The agent tried to run the full RAGAS evaluation but encountered an error. Check logs for details."

    # 2) Run simplified evaluation
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

    # 3) Create a note in rag_db.notes
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

    # 4) Inspect raw documents from rag_db.passages
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

    # 5) Rebuild FAISS index
    if mode == "rebuild_index":
        try:
            info = rebuild_full_index()
            return (
                "I triggered a full rebuild of the FAISS index "
                f"with batch_size={info.get('batch_size')}."
            )
        except Exception as e:
            rag_windows.log(f"[agent] Rebuild_index mode failed: {e}")
            return "The agent tried to rebuild the FAISS index but failed."

    # 6) Tag passages for a query
    if mode == "tag_passages":
        try:
            # Heuristic: use the whole goal as the query and a simple tag
            tag = "important"
            info = tag_top_passages(goal, tag=tag, k=max(1, k))
            return (
                f"I tagged {info.get('modified', 0)} passages (matched {info.get('matched', 0)}) "
                f"with tag '{tag}' for this goal."
            )
        except Exception as e:
            rag_windows.log(f"[agent] Tag_passages mode failed: {e}")
            return "The agent tried to tag passages in MongoDB but failed."

    # 7) Manage evaluation samples
    if mode == "manage_eval":
        try:
            info = add_eval_sample_from_goal(goal, k=max(1, k))
            return (
                "I created an evaluation sample in rag_db.eval_samples "
                f"with id {info.get('inserted_id')} using {info.get('num_contexts', 0)} contexts."
            )
        except Exception as e:
            rag_windows.log(f"[agent] Manage_eval mode failed: {e}")
            return "The agent tried to create an evaluation sample but failed."

    # 8) Planned multi-step: currently a thin wrapper that explains capabilities
    #    and then performs a search+summarize as a demo step.
    if mode == "planned":
        # Reuse the LLM to describe what tools exist and then do search+summarize.
        system_expl = (
            "You are an assistant orchestrating a local RAG agent. "
            "The agent has tools: search_docs (semantic search over passages), "
            "tag_top_passages (label top-k results), rebuild_full_index (rebuild FAISS index), "
            "and add_eval_sample_from_goal (store a question+contexts for later evaluation). "
            "For now, you will briefly explain how you would use 2-3 steps with these tools "
            "to accomplish the user's goal, then actually call search_docs implicitly and "
            "summarize the results."
        )

        rag_windows.log("[agent] Running planned multi-step helper (single-LLM turn + search)")
        # Step 1: retrieve passages
        passages = search_docs(goal, k=max(1, k))
        if not passages:
            return (
                "If I had relevant passages, I would first run search_docs to find them, "
                "then potentially tag important ones and save an evaluation sample. "
                "However, no passages were retrieved for this goal."
            )

        # Step 2: let the LLM both plan and summarize based on retrieved passages
        llm = rag_windows._load_llm(mp)
        context_blocks: List[str] = []
        for p in passages:
            sid = p.get("source_id", "?")
            text = p.get("text", "")
            preview = text if len(text) <= 600 else text[:600] + "..."
            context_blocks.append(f"[{sid}]\n{preview}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            system_expl
            + "\n\nUSER GOAL:\n" + goal
            + "\n\nRETRIEVED PASSAGES:\n" + context_text
            + "\n\nIn 2-3 short paragraphs, first describe a reasonable 2-3 step "
              "tool plan using the available tools, then provide the final answer "
              "to the goal based only on the passages."
            + "\n\nPLAN AND ANSWER:"
        )

        try:
            out = llm.create_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=0.2,
                top_p=0.9,
            )
            return out["choices"][0]["text"].strip()
        except Exception as e:
            rag_windows.log(f"[agent] Planned mode failed, falling back to search_summarize: {e}")
            # Fall back to plain search+summarize below.

    # 9) Default: search + summarize
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