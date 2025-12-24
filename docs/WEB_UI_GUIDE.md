# Web UI Guide

This guide explains how to use the React-based web interface for the RAG system.

## Overview

The web UI provides a user-friendly interface for:
- **Ingesting documents** (TXT, PDF, DOCX, CSV, XLSX, JSON, XML, images with OCR)
- **Building FAISS indexes**
- **Querying the RAG system** with natural language questions
- **Retrieving passages** without generation (for testing)

The web app runs on **React** and connects to the FastAPI backend.

---

## Quick Start

### 1. Start the FastAPI Backend

In PowerShell, from the project root:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Recommended CPU defaults (optional):
#   $env:LLM_N_CTX=2048; $env:LLM_MAX_TOKENS=128; $env:LLM_N_BATCH=256

# Start API server on port 8000
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
```

Keep this terminal running. The API will be available at `http://localhost:8000`.

### 2. Install Node.js Dependencies

Open a **new PowerShell terminal**, navigate to the `web/` folder:

```powershell
cd web
npm install
```

This installs React and all dependencies listed in `package.json`.

### 3. Start the React Development Server

```powershell
npm start
```

This will:
- Start the React dev server on `http://localhost:3000`
- Automatically open your default browser
- Enable hot-reload (changes update instantly)

---

## Using the Web Interface

### Status Bar

The top bar shows:
- **API Server Status**: Green dot = online, red dot = offline
- **Refresh Status** button: Manually check API health

### Tab 1: Ingest Files

1. Click the **upload area** or drag files into it
2. Select one or more files (TXT, PDF, DOCX, CSV, XLSX, JSON, XML, PNG/JPG)
3. Review selected files in the list
4. Click **Ingest Files** to upload and process
5. Success message shows how many passages were ingested

**Supported Formats:**
- Text: `.txt`
- PDF: `.pdf` (requires PyPDF2)
- Word: `.docx` (requires python-docx)
- Excel: `.xlsx` (requires pandas)
- CSV: `.csv` (requires pandas)
- JSON: `.json`
- XML: `.xml`
- Images: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` (OCR with Tesseract)

### Tab 2: Build Index

1. Click **Build Index**
2. Wait for FAISS index creation (progress shown)
3. Success message shows index size (number of vectors)

**Note:** Always build the index **after** ingesting new documents.

### Tab 3: Query RAG

1. Enter your question in the text area
2. Adjust **k** (number of passages to retrieve, default: 3)
3. Click **Generate Answer**
4. View:
   - **Generated Answer**: LLM response with inline citations
   - **Retrieved Passages**: Source passages used to generate the answer

**Example Questions:**
- "What is machine learning?"
- "Explain neural networks"
- "What are the key concepts in deep learning?"

### Tab 4: Retrieve Only

Same as Query RAG, but **skips LLM generation**.

Useful for:
- Testing retrieval quality
- Debugging without waiting for generation
- Checking source passage relevance

---

## Architecture

```
┌─────────────┐         HTTP          ┌─────────────┐
│             │  ──────────────────►  │             │
│  React App  │                       │  FastAPI    │
│  (Port 3000)│  ◄──────────────────  │  (Port 8000)│
└─────────────┘      JSON/REST        └──────┬──────┘
                                              │
                                              ├─► MongoDB
                                              ├─► FAISS Index
                                              └─► Llama.cpp
```

### Request Flow

1. **User Action** → React component
2. **API Call** → `apiService.js` sends fetch request
3. **FastAPI** → Processes request (calls `rag_windows.py` functions)
4. **Backend** → Interacts with MongoDB, FAISS, Llama.cpp
5. **Response** → JSON returned to React
6. **UI Update** → Results displayed in styled components

---

## Configuration

### Change API Port

If you need to run the API on a different port:

1. **Start API with custom port:**
   ```powershell
   .\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8080
   ```

2. **Update `web/src/apiService.js`:**
   ```javascript
   const API_BASE_URL = 'http://localhost:8080';
   ```

3. **Update `web/package.json` proxy:**
   ```json
   "proxy": "http://localhost:8080"
   ```

### CORS Configuration

CORS is pre-configured in `src/api.py` to allow requests from `http://localhost:3000`.

If you deploy to production, update the `allow_origins` list:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://yourdomain.com"
    ],
    ...
)
```

---

## Production Deployment

### Build React App for Production

```powershell
cd web
npm run build
```

This creates an optimized production build in `web/build/`.

### Serve Static Build

You can serve the build folder with the FastAPI backend:

1. **Install static files support:**
   ```powershell
   .\.venv\Scripts\python.exe -m pip install aiofiles
   ```

2. **Add static mount to `src/api.py`:**
   ```python
   from fastapi.staticfiles import StaticFiles
   
   app.mount("/", StaticFiles(directory="web/build", html=True), name="static")
   ```

3. **Start server:**
   ```powershell
   .\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
   ```

4. **Access UI at:** `http://localhost:8000`

---

## Troubleshooting

### API Status Shows "Offline"

**Check:**
1. Is the FastAPI server running? (`uvicorn src.api:app ...`)
2. Is MongoDB running? (`mongod --dbpath ...`)
3. Is the API port correct? (default: 8000)
4. Check browser console for CORS errors

**Fix:**
- Start API server: `.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000`
- Start MongoDB: See `docs/PROJECT_GUIDE.md`

### File Upload Fails

**Check:**
1. File format is supported (see list above)
2. Optional dependencies are installed (e.g., PyPDF2 for PDF)
3. API server logs for errors

**Fix:**
- Install missing dependencies: See `docs/INGESTION_MODULE.md`
- Check API terminal for error messages

### Query Takes Too Long

**LLM generation is CPU-intensive on Windows.**

**Optimize:**
- Use smaller models (TinyLlama 1.1B is default)
- Reduce `k` (number of passages) to 2-3
- Use **Retrieve Only** tab for instant results (no LLM)

### Build Index Fails

**Check:**
1. Have you ingested documents?
2. Is MongoDB accessible?
3. Are embeddings dependencies installed? (sentence-transformers)

**Fix:**
```powershell
# Check MongoDB passages
.\.venv\Scripts\python.exe src\rag_windows.py --retrieve-only --query "test"

# If no passages, ingest files first
.\.venv\Scripts\python.exe src\rag_windows.py --ingest-files
```

### React Dev Server Won't Start

**Error: `npm: command not found`**

**Fix:** Install Node.js from [nodejs.org](https://nodejs.org/)

**Error: `Port 3000 already in use`**

**Fix:**
```powershell
# Find process using port 3000
netstat -ano | findstr :3000

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use a different port
$env:PORT=3001; npm start
```

---

## Advanced Features

### Custom Embedding Models

To use a different SentenceTransformers model:

1. Update `src/rag_windows.py`:
   ```python
   model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
   ```

2. Rebuild index with the new model

### Custom LLM Models

To use a different GGUF model:

1. Download a `.gguf` file to `models/`
2. Update `src/rag_windows.py`:
   ```python
   model_path = Path(__file__).parent.parent / "models" / "your-model.gguf"
   ```

3. Restart API server

### Add Database Ingestion to Web UI

The current web UI only supports file uploads. To add Oracle/MongoDB ingestion:

1. Add new tabs in `web/src/App.js`
2. Add form inputs for connection strings
3. Call `apiService.ingestOracle()` or `apiService.ingestMongo()`

Example API calls are already in `apiService.js` but not exposed in the UI.

---

## Next Steps

- **Explore the API:** Visit `http://localhost:8000/docs` for Swagger UI
- **Read API Guide:** See `docs/API_GUIDE.md` for REST API details
- **Read Project Guide:** See `docs/PROJECT_GUIDE.md` for architecture details
- **Customize UI:** Edit `web/src/App.css` for styling changes

---

**Need Help?**
- Check `docs/PROJECT_GUIDE.md` for architecture details
- Check `docs/INGESTION_MODULE.md` for file format support
- Check `docs/API_GUIDE.md` for REST API reference
