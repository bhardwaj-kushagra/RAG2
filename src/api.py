"""
FastAPI server exposing RAG pipeline endpoints.

Endpoints:
- POST /ingest/files       : Ingest files from data/docs/
- POST /ingest/oracle      : Ingest from Oracle DB
- POST /ingest/mongo       : Ingest from external MongoDB
- POST /build-index        : Build FAISS index
- POST /retrieve           : Retrieve top-K passages (no generation)
- POST /query              : Full RAG query with generation
- POST /agent              : Run agentic AI over RAG + DB tools
- GET  /health             : Health check
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent to path to import rag_windows
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import core functions from rag_windows
from rag_windows import (
    ingest_files,
    ingest_oracle,
    ingest_mongo_source,
    build_index as build_faiss_index,
    retrieve,
    generate_answer,
    get_collection,
    Passage,
    DEFAULT_MODEL_PATH,
)
import agent as agent_module

app = FastAPI(
    title="Local RAG API",
    description="REST API for ingestion, indexing, retrieval, and generation using MongoDB, FAISS, and llama.cpp",
    version="1.0.0",
)


# --- Request/Response Models ---


class IngestFilesRequest(BaseModel):
    pass  # Uses data/docs/ directory


class IngestOracleRequest(BaseModel):
    dsn: str = Field(..., description="Oracle DSN (host:port/service)")
    user: str = Field(..., description="Oracle username")
    password: str = Field(..., description="Oracle password")
    query: str = Field(
        default="SELECT ID, TITLE, BODY FROM DOCUMENTS",
        description="SQL query returning ID, TITLE, BODY",
    )


class IngestMongoRequest(BaseModel):
    uri: str = Field(..., description="Source MongoDB URI")
    db_name: str = Field(..., description="Source database name")
    collection: str = Field(..., description="Source collection name")
    text_field: str = Field(default="text", description="Field containing text")
    id_field: Optional[str] = Field(default=None, description="Optional ID field")


class BuildIndexRequest(BaseModel):
    batch_size: int = Field(default=64, description="Batch size for embedding")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of passages to retrieve")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Question text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of passages to retrieve")
    model_path: Optional[str] = Field(default=None, description="Path to .gguf model (optional)")


class AgentRequest(BaseModel):
    goal: str = Field(..., description="High-level agent goal (e.g., 'Run a quick RAG evaluation', 'Summarize docs about X')")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of passages to retrieve in search_summarize mode")
    model_path: Optional[str] = Field(default=None, description="Path to .gguf model (optional)")


class AgentResponse(BaseModel):
    goal: str
    mode: str
    answer: str


class PassageResponse(BaseModel):
    source_id: str
    text: str


class RetrieveResponse(BaseModel):
    query: str
    passages: List[PassageResponse]


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]


class StatusResponse(BaseModel):
    message: str
    details: Optional[Dict[str, Any]] = None


# Configure CORS to allow React app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---


@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint."""
    try:
        coll = get_collection()
        count = coll.count_documents({})
        return StatusResponse(
            message="Healthy",
            details={"passages_count": count},
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/ingest/files", response_model=StatusResponse)
async def api_ingest_files():
    """Ingest all supported files from data/docs/ directory."""
    try:
        ingest_files()
        coll = get_collection()
        count = coll.count_documents({})
        return StatusResponse(
            message="Files ingested successfully",
            details={"total_passages": count},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/ingest/upload", response_model=StatusResponse)
async def api_upload_and_ingest(file: UploadFile = File(...)):
    """Upload a file and ingest it immediately."""
    try:
        # Ensure data/docs exists
        docs_dir = Path("data/docs")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded file
        file_path = docs_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Trigger ingestion (this will scan the folder, picking up the new file)
        # In a more advanced setup, we might want to ingest just this file,
        # but ingest_files() is idempotent-ish (checks hashes/existence usually) or just re-ingests.
        # Given our current simple implementation, it re-scans.
        ingest_files()
        
        coll = get_collection()
        count = coll.count_documents({})
        
        return StatusResponse(
            message=f"File '{file.filename}' uploaded and ingested successfully",
            details={"total_passages": count},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload/Ingestion failed: {e}")


@app.post("/ingest/oracle", response_model=StatusResponse)
async def api_ingest_oracle(req: IngestOracleRequest):
    """Ingest rows from an Oracle database table."""
    try:
        ingest_oracle(req.dsn, req.user, req.password, req.query)
        coll = get_collection()
        count = coll.count_documents({})
        return StatusResponse(
            message="Oracle data ingested successfully",
            details={"total_passages": count},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Oracle ingestion failed: {e}")


@app.post("/ingest/mongo", response_model=StatusResponse)
async def api_ingest_mongo(req: IngestMongoRequest):
    """Ingest documents from an external MongoDB collection."""
    try:
        ingest_mongo_source(
            uri=req.uri,
            db_name=req.db_name,
            collection=req.collection,
            text_field=req.text_field,
            id_field=req.id_field,
        )
        coll = get_collection()
        count = coll.count_documents({})
        return StatusResponse(
            message="MongoDB source ingested successfully",
            details={"total_passages": count},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB ingestion failed: {e}")


@app.post("/build-index", response_model=StatusResponse)
async def api_build_index(req: BuildIndexRequest):
    """Build FAISS index from passages stored in MongoDB."""
    try:
        build_faiss_index(batch_size=req.batch_size)
        return StatusResponse(message="FAISS index built successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index build failed: {e}")


@app.post("/retrieve", response_model=RetrieveResponse)
async def api_retrieve(req: RetrieveRequest):
    """Retrieve top-K passages for a query (no generation)."""
    try:
        passages = retrieve(req.query, top_k=req.top_k)
        return RetrieveResponse(
            query=req.query,
            passages=[PassageResponse(source_id=p.source_id, text=p.text) for p in passages],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")


@app.post("/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    """Full RAG query: retrieve passages and generate an answer."""
    try:
        passages = retrieve(req.query, top_k=req.top_k)
        if not passages:
            raise HTTPException(status_code=404, detail="No relevant passages found")

        model_path = Path(req.model_path) if req.model_path else DEFAULT_MODEL_PATH
        # Use non-streaming generation for HTTP API
        answer = generate_answer(req.query, passages, model_path)

        # Extract sources
        sources = list({p.source_id for p in passages})

        return QueryResponse(
            query=req.query,
            answer=answer,
            sources=sources,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.post("/agent", response_model=AgentResponse)
async def api_agent(req: AgentRequest):
    """Run the lightweight agent over RAG + DB tools.

    This wraps agent.run_agent so the web UI (and other clients)
    can trigger evaluation, note-taking, DB inspection, or
    search+summarize flows.
    """
    goal = req.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="Agent goal must not be empty")

    try:
        model_path = Path(req.model_path) if req.model_path else DEFAULT_MODEL_PATH
        # agent.run_agent already logs the detected mode
        answer = agent_module.run_agent(goal=goal, model_path=model_path, k=req.top_k)

        # For now, we don't expose the exact mode from agent internals,
        # so we approximate from the text; this can be refined later.
        lower = goal.lower()
        if any(kw in lower for kw in ["ragas", "evaluation", "rag quality", "run evaluation"]):
            mode = "evaluation"
        elif any(kw in lower for kw in ["note", "todo", "task", "remind", "remember"]):
            mode = "note"
        elif any(kw in lower for kw in ["inspect db", "inspect database", "raw db", "raw documents", "rag_db.passages"]):
            mode = "inspect_db"
        else:
            mode = "search_summarize"

        return AgentResponse(goal=goal, mode=mode, answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent request failed: {e}")


# --- Run Server ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
