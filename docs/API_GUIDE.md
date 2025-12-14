# FastAPI RAG Server - Quick Start & Usage Guide

REST API exposing the RAG pipeline for ingestion, indexing, retrieval, and generation.

## Installation

```powershell
# Already installed if you followed main setup
pip install fastapi uvicorn python-multipart
```

## Start the Server

```powershell
# From project root
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000

# Or with reload (development)
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000 --reload
```

Server will be available at: `http://localhost:8000`

Interactive API docs (Swagger): `http://localhost:8000/docs`

ReDoc documentation: `http://localhost:8000/redoc`

## API Endpoints

### Health Check
```powershell
curl http://localhost:8000/health
```

### Ingest Files
Ingest all supported files from `data/docs/`:
```powershell
Invoke-RestMethod -Uri http://localhost:8000/ingest/files -Method POST
```

```powershell
$uri = "http://localhost:8000/ingest/upload"
$filePath = "C:\path\to\your\file.pdf"
Invoke-RestMethod -Uri $uri -Method Post -InFile $filePath -ContentType "multipart/form-data"
```


### Build Index
Build FAISS index from MongoDB passages:
```powershell
Invoke-RestMethod -Uri http://localhost:8000/build-index `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"batch_size": 64}'
```

### Retrieve Passages (No Generation)
```powershell
Invoke-RestMethod -Uri http://localhost:8000/retrieve `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query": "What is machine learning?", "top_k": 5}'
```

### Full RAG Query (with Generation)
```powershell
Invoke-RestMethod -Uri http://localhost:8000/query `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query": "What is machine learning?", "top_k": 5}'
```

### Ingest from Oracle
```powershell
$body = @{
    dsn = "host:port/service"
    user = "username"
    password = "password"
    query = "SELECT ID, TITLE, BODY FROM DOCUMENTS"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/ingest/oracle `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

### Ingest from External MongoDB
```powershell
$body = @{
    uri = "mongodb://localhost:27017"
    db_name = "source_db"
    collection = "articles"
    text_field = "content"
    id_field = "slug"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/ingest/mongo `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

## Request/Response Examples

### Retrieve Response
```json
{
  "query": "What is machine learning?",
  "passages": [
    {
      "source_id": "test_ml.txt#0",
      "text": "Machine learning is a subset of artificial intelligence..."
    }
  ]
}
```

### Query Response
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of AI that enables systems to learn from data...",
  "sources": ["test_ml.txt#0", "test_deeplearning.docx#0"]
}
```

## Python Client Example

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Ingest files
response = requests.post("http://localhost:8000/ingest/files")
print(response.json())

# Build index
response = requests.post(
    "http://localhost:8000/build-index",
    json={"batch_size": 64}
)
print(response.json())

# Query
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is RAG?",
        "top_k": 5
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## JavaScript/Node.js Client Example

```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(console.log);

// Query
fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: 'What is RAG?',
    top_k: 5
  })
})
.then(r => r.json())
.then(data => {
  console.log('Answer:', data.answer);
  console.log('Sources:', data.sources);
});
```

## Error Handling

All endpoints return standard HTTP status codes:
- `200`: Success
- `404`: Resource not found (e.g., no passages retrieved)
- `500`: Server error
- `503`: Service unhealthy (e.g., MongoDB connection failed)

Error response format:
```json
{
  "detail": "Error message description"
}
```

## Production Deployment

For production use:

```powershell
# With more workers
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000 --workers 4

# With HTTPS (requires cert files)
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 443 `
  --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

## Notes

- CLI (`rag_windows.py`) and API can be used simultaneously
- Both share the same MongoDB collection and FAISS index
- API does not support file upload directly; place files in `data/docs/` and call `/ingest/files`
- For large-scale deployments, consider adding authentication, rate limiting, and caching

## Troubleshooting

- **Server won't start**: Check if port 8000 is already in use
- **Import errors**: Ensure all dependencies are installed in the venv
- **MongoDB errors**: Verify MongoDB is running locally
- **Generation timeout**: Increase uvicorn timeout with `--timeout-keep-alive 300`

## OpenAPI Schema

Access the full OpenAPI schema at: `http://localhost:8000/openapi.json`
