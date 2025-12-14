# Agentic AI Guide

This document describes the lightweight agentic AI capabilities in this project,
how to use them from both the command-line interface (CLI) and the React web UI,
plus notes on limitations and future options.

## Overview

The agent is a small, rule-based layer on top of the existing RAG pipeline and
MongoDB data store. It does **not** use complex tool-calling protocols; instead,
it maps your goal text onto a few concrete modes and then calls the appropriate
Python functions.

Core implementation files:
- `src/agent.py` – agent logic and tools
- `src/rag_windows.py` – CLI entrypoint and RAG utilities
- `src/api.py` – FastAPI server exposing `/agent`
- `web/src` – React web UI, including the Agentic AI tab

## Current Agent Capabilities

The agent supports four main modes, chosen via simple keyword heuristics over
your goal text:

1. **Evaluation mode (`evaluation`)**
   - Triggered by words like: `ragas`, `evaluation`, `evaluate quality`,
     `rag quality`, `run evaluation`.
   - Action:
     - Calls `evaluation.run_evaluation_simple(...)` to run the simplified
       RAG evaluation using embedding-based metrics.
     - Returns aggregate scores:
       - Faithfulness
       - Answer relevancy
       - Context precision
       - Context recall
       - Number of samples evaluated
   - Best used for quick, local quality checks of your RAG setup.

2. **Note mode (`note`)**
   - Triggered by words like: `note`, `todo`, `task`, `remind`, `remember`,
     `create a note`, `add a note`.
   - Action:
     - Inserts a document into `rag_db.notes` with the goal text and simple
       metadata (e.g., `created_by: agent`, `kind: note`).
   - Use it as a lightweight way to store reminders or design notes tied to
     your RAG project.

3. **Inspect DB mode (`inspect_db`)**
   - Triggered by phrases like: `inspect db`, `inspect database`, `raw db`,
     `raw documents`, `what is in rag_db.passages`, `show passages`.
   - Action:
     - Samples up to 20 documents from `rag_db.passages` and builds a prompt
       with JSON-like previews (IDs, sources, truncated text, timestamps).
     - Asks the LLM to summarize what kind of data is stored there and how it
       is used in the RAG pipeline (3–5 sentences).
   - Helps you understand and debug what has actually been ingested.

4. **Search + Summarize mode (`search_summarize`)**
   - Default when none of the above mode keywords match.
   - Action:
     - Calls `search_docs(...)`, which uses FAISS retrieval over `rag_db.passages`.
     - Builds a prompt with the retrieved passages and your goal.
     - Asks the LLM for a concise (2–4 sentence) answer based **only** on those
       passages.
   - This is a more agentic way of doing a normal RAG query: you give a high-
     level goal and let the agent decide to search and summarize.

### Tools Under the Hood

The agent uses small tool functions defined in `src/agent.py`:

- `search_docs(query, k)` – wraps `rag_windows.retrieve(...)`.
- `get_raw_from_db(filter)` – reads directly from `rag_db.passages`.
- `write_note_to_db(text, meta)` – writes to `rag_db.notes`.

The LLM used by the agent is the same local GGUF model as the main RAG pipeline
(default: Meta-Llama-3.1-8B-Instruct Q4_K_M in `models/`).

## CLI Usage

The CLI entrypoint for the agent is `src/rag_windows.py` via the `--agent` flag.

### Basic Commands

Run a quick evaluation of RAG quality:

```bash
python src/rag_windows.py --agent "Run a quick evaluation of our RAG quality."
```

Create a note in `rag_db.notes`:

```bash
python src/rag_windows.py --agent "Create a note to revisit the HelioScope evaluation tomorrow."
```

Inspect what is stored in `rag_db.passages`:

```bash
python src/rag_windows.py --agent "Inspect what kind of documents are stored in rag_db.passages."
```

Search and summarize (default mode):

```bash
python src/rag_windows.py --agent "Summarize what we know about Dr. Aria Khatri and the HelioScope project."
```

You can also override the model path if needed:

```bash
python src/rag_windows.py --agent "Run a quick evaluation" \
  --model-path models/meta-llama-3.1-8b-instruct-q4_k_m.gguf
```

CLI behavior:
- The script prints the classified mode in the logs.
- For evaluation mode, it prints a short, human-readable summary of metrics.
- For other modes, it prints the LLM-generated answer or a short status message.

## Web UI Usage

The React web app now exposes the agent via a dedicated **Agentic AI** tab.

### Where to find it

- File: `web/src/App.js`.
- Tab label: **🧠 Agentic AI**.

### How to use it

1. Start the API server (from the project root):

   ```bash
   python src/api.py
   ```

2. Start the React dev server in the `web/` folder:

   ```bash
   cd web
   npm install   # if not already done
   npm start
   ```

3. In the browser (http://localhost:3000):
   - Ensure the status bar shows **API Server: ✓ Online**.
   - Click the **🧠 Agentic AI** tab.

4. Fill in the **Agent Goal** textarea, for example:
   - `Run a quick evaluation of our RAG quality.`
   - `Summarize what we know about HelioScope.`
   - `Inspect what is stored in rag_db.passages.`

5. Optionally adjust **Passages to retrieve (k)** for search/summarize mode.

6. Click **Run Agent**.

7. The result panel shows:
   - **Mode** – one of `evaluation`, `note`, `inspect_db`, `search_summarize`
     (inferred heuristically from your goal).
   - **Agent Result** – the natural-language answer or summary.

Note:
- Evaluation mode uses only the backend evaluation logic (no passages are
  displayed in the UI, just the metric summary in text form).
- For search/summarize mode, the agent uses the same FAISS + Mongo pipeline
  as the standard RAG queries.

## Usage Guidelines

- Prefer **clear, explicit goals**:
  - Good: "Run a quick evaluation of our RAG quality."
  - Good: "Inspect what kind of documents are in rag_db.passages."
  - Ambiguous: "Do something smart" (falls back to search_summarize).

- Use the agent for:
  - Lightweight quality checks (evaluation mode).
  - Quick meta-operations on your data (inspect_db mode).
  - High-level queries where you want the system to both search and summarize.

- Continue to use the standard **Query RAG** and **Retrieve Only** tabs when you
  want tight control over the exact question and retrieved passages.

## Limitations and Future Options

### Current limitations

- **Rule-based mode selection** – the agent uses simple keyword checks. It does
  not yet use the LLM itself to choose tools or build multi-step plans.
- **Single-step flows** – each call runs a single pass (evaluate, note, inspect,
  or search+summarize). There is no memory of prior agent runs beyond what you
  explicitly store in `rag_db.notes`.
- **No streaming in web UI** – CLI generation supports streaming tokens, but
  the web API currently returns full answers only after generation completes.
- **Limited error surfacing** – database or model errors are surfaced as simple
  error messages; there is no rich per-step tracing yet.

### Future enhancements (ideas)

- **Richer tool set**:
  - Tools for re-indexing specific sources.
  - Tools for managing evaluation datasets.
  - Tools for editing or tagging passages directly.

- **LLM-based planning**:
  - Let the LLM choose among tools (search, evaluate, inspect, note) given a
    natural-language goal, instead of fixed keyword rules.
  - Multi-step workflows (e.g., run evaluation, summarize results, and log
    a note automatically).

- **Web UI improvements**:
  - Show retrieved passages alongside agent answers for search/summarize mode.
  - Progress indicators for longer-running evaluations.
  - Optional streaming of tokens over WebSocket or Server-Sent Events.

- **Advanced evaluation**:
  - Integrate full RAGAS with OpenAI or other hosted models as an optional,
    opt-in path for deeper evaluations when internet access is available.

This guide should give you a practical understanding of what the agent can do
right now, how to drive it from both CLI and web UI, and where it can be
extended next.