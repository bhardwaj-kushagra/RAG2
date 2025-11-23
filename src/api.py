"""
FastAPI server exposing RAG pipeline endpoints.

Endpoints:
- POST /ingest/files       : Ingest files from data/docs/
- POST /ingest/oracle      : Ingest from Oracle DB
- POST /ingest/mongo       : Ingest from external MongoDB
- POST /build-index        : Build FAISS index
- POST /retrieve           : Retrieve top-K passages (no generation)
- POST /query              : Full RAG query with generation
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


# --- Run Server ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
