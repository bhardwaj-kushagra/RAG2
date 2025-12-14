# RAG System - Complete Overview

## What You Have

A **complete, production-ready RAG system** with three interfaces:

### 1. ✅ CLI Interface (src/rag_windows.py)
Command-line tool for advanced users and automation.

```powershell
# Ingest documents
python src/rag_windows.py --ingest-files

# Build index
python src/rag_windows.py --build-index

# Query
python src/rag_windows.py --query "What is machine learning?"
```

### 2. ✅ REST API (src/api.py)
FastAPI server for programmatic access and integration.

```powershell
# Start server
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000

# Access Swagger UI
# http://localhost:8000/docs
```

### 3. ✅ Web UI (web/)
**NEW!** Beautiful React interface for demos and presentations.

```powershell
# Option A: Use the start script (launches both servers)
.\start_web.ps1

# Option B: Manual start (2 terminals)
# Terminal 1: API
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000

# Terminal 2: React
cd web
npm start
```

Open browser to: **http://localhost:3000**

---

## Features of the Web UI

### 🎨 Professional Design
- Modern gradient design with smooth animations
- Tab-based navigation for clear workflow
- Real-time API status indicator
- Responsive layout

### 📁 File Upload Tab
- Drag-and-drop file upload
- Multi-file selection
- Supports 11+ file formats:
  - Text: TXT
  - Documents: PDF, DOCX
  - Data: CSV, XLSX, JSON, XML
  - Images: PNG, JPG, JPEG, TIF, TIFF (with OCR)
- Progress feedback
- File list preview

### 🔍 Build Index Tab
- One-click index building
- Progress indication
- Index size display
- MongoDB connection status

### 💬 Query RAG Tab
- Natural language question input
- Adjustable K (number of passages to retrieve)
- **Generated answer** with inline citations
- **Source passages** displayed with highlighting
- Loading spinner during processing

### 📚 Retrieve Only Tab
- Test retrieval without LLM generation
- Faster than full RAG query
- Shows retrieved passages with source IDs
- Useful for debugging retrieval quality

### ⚡ Real-time Features
- API health monitoring
- Auto-refresh status
- Error handling with user-friendly messages
- Success/error visual feedback

---

## Quick Start Guide

### First Time Setup

1. **Install Node.js** (if not installed)
   - Download from: https://nodejs.org/
   - Verify: `node --version`

2. **Run setup script**
   ```powershell
   .\setup_web.ps1
   ```
   This checks all dependencies and installs React packages.

3. **Start MongoDB**
   ```powershell
   # See docs/PROJECT_GUIDE.md for MongoDB setup
   mongod --dbpath "C:\data\db"
   ```

4. **Start the web UI**
   ```powershell
   .\start_web.ps1
   ```
   Opens browser automatically to http://localhost:3000

### Daily Use

```powershell
# One command to start everything
.\start_web.ps1

# Press Ctrl+C to stop both servers
```

---

## Project Structure

```
RAG/
├── src/
│   ├── rag_windows.py      # Core RAG logic (CLI)
│   └── api.py              # FastAPI server (REST API)
│
├── web/                     # React web UI
│   ├── src/
│   │   ├── App.js          # Main UI component
│   │   ├── App.css         # Styling
│   │   ├── apiService.js   # API client
│   │   └── index.js        # React entry
│   ├── public/
│   │   └── index.html      # HTML template
│   ├── package.json        # Dependencies
│   └── README.md           # Web-specific docs
│
├── docs/
│   ├── PROJECT_GUIDE.md     # Full architecture guide
│   ├── INGESTION_MODULE.md  # Multi-format ingestion
│   ├── API_GUIDE.md         # REST API reference
│   └── WEB_UI_GUIDE.md      # Web UI detailed guide
│
├── data/docs/              # Input documents
├── models/                 # LLM models (.gguf)
│
├── setup_web.ps1          # One-time web setup
├── start_web.ps1          # Start both servers
├── .gitignore             # Git exclusions
└── README.md              # Quick start
```

---

## Technology Stack

### Backend
- **Python 3.11**
- **MongoDB** - Document storage
- **SentenceTransformers** - Embeddings (all-MiniLM-L6-v2)
- **FAISS** - Vector similarity search
- **llama-cpp-python** - Local LLM inference (TinyLlama 1.1B)
- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server

### Frontend
- **React 18** - UI framework
- **React Scripts 5** - Build tooling
- **Fetch API** - HTTP client
- **CSS3** - Styling with animations

### Optional Dependencies
- **PyPDF2** - PDF extraction
- **python-docx** - DOCX extraction
- **pandas** - Excel/CSV processing
- **pytesseract** - OCR
- **pdf2image** - PDF to image conversion
- **oracledb** - Oracle DB connection
- **pillow** - Image processing

---

## Documentation Map

1. **New to RAG?** → Read `docs/PROJECT_GUIDE.md`
   - Explains RAG architecture
   - Setup instructions
   - How the pipeline works

2. **Want to ingest different file types?** → Read `docs/INGESTION_MODULE.md`
   - Supported formats
   - Optional dependencies
   - OCR configuration
   - Database ingestion

3. **Building an integration?** → Read `docs/API_GUIDE.md`
   - REST API endpoints
   - Request/response schemas
   - Client examples (Python, JavaScript, PowerShell)

4. **Using the web interface?** → Read `docs/WEB_UI_GUIDE.md`
   - Web UI features
   - Setup and configuration
   - Troubleshooting
   - Production deployment

5. **Quick reference** → Read `README.md` (this file)

---

## Usage Examples

### Demo Scenario 1: File Upload and Query

1. Open http://localhost:3000
2. Click **"Ingest Files"** tab
3. Drag-and-drop your documents (PDFs, DOCX, TXT, etc.)
4. Click **"Ingest Files"**
5. Click **"Build Index"** tab → **"Build Index"** button
6. Click **"Query RAG"** tab
7. Type: "What are the main topics in these documents?"
8. Click **"Generate Answer"**
9. View AI-generated answer with source citations!

### Demo Scenario 2: Retrieve Without Generation

1. Click **"Retrieve Only"** tab
2. Enter question: "machine learning algorithms"
3. Set K = 5
4. Click **"Retrieve Passages"**
5. See top 5 relevant passages instantly (no LLM wait)

### Demo Scenario 3: API Integration

```python
import requests

# Upload file
files = {'files': open('report.pdf', 'rb')}
response = requests.post('http://localhost:8000/ingest/files', files=files)
print(response.json())

# Build index
response = requests.post('http://localhost:8000/build-index', 
                        json={'embedding_model': 'all-MiniLM-L6-v2'})
print(response.json())

# Query
response = requests.post('http://localhost:8000/query',
                        json={'query': 'What is the summary?', 'top_k': 3})
print(response.json()['answer'])
```

---

## Performance Tips

### For Faster Queries
- Use **Retrieve Only** tab (skips LLM generation)
- Reduce K to 2-3 passages
- Use smaller LLM models (TinyLlama 1.1B is default)

### For Better Answers
- Increase K to 5-10 passages
- Use larger LLM models (e.g., Llama 2 7B)
- Ensure high-quality source documents
- Build index after ingesting all documents

### For Scaling
- Index MongoDB with `db.passages.createIndex({source_id: 1})`
- Use MongoDB Atlas for cloud storage
- Deploy API with Gunicorn/multiple workers
- Use production React build (`npm run build`)

---

## Troubleshooting

### Web UI shows "API Offline"
- Check if MongoDB is running
- Check if API server is running (port 8000)
- Verify CORS is enabled in `src/api.py`

### File upload fails
- Check file format is supported
- Install optional dependencies (see `docs/INGESTION_MODULE.md`)
- Check API server logs for errors

### Generation is slow
- **Normal!** CPU-only LLM inference is slow
- Use smaller models (TinyLlama)
- Use "Retrieve Only" tab for instant results

### npm install fails
```powershell
npm cache clean --force
rm -rf web/node_modules web/package-lock.json
cd web
npm install
```

### Port conflicts
```powershell
# Change API port
.\.venv\Scripts\uvicorn.exe src.api:app --port 8080

# Change React port
cd web
$env:PORT=3001; npm start
```

---

## Next Steps

### Extend the System
- Add more extractors (HTML, Markdown, etc.)
- Implement chunking strategies (recursive, semantic)
- Add metadata filtering
- Implement reranking with cross-encoder
- Add user authentication
- Add chat history/memory

### Production Deployment
- Build React for production: `npm run build`
- Serve with FastAPI static files
- Use reverse proxy (Nginx)
- Add SSL/TLS certificates
- Monitor with logging/metrics
- Scale with load balancer

### Integrate with Other Tools
- Jupyter notebooks (API client)
- VS Code extensions
- Excel add-ins
- PowerBI dashboards
- Slack/Discord bots

---

## Community & Support

This is an educational project demonstrating:
- ✅ Local-first RAG (no cloud APIs)
- ✅ Multi-format document ingestion
- ✅ Production-ready API design
- ✅ Modern web UI with React
- ✅ Windows-friendly setup

**Contributions welcome!**
- Improve extraction quality
- Add new file formats
- Optimize performance
- Enhance UI/UX
- Write more documentation

---

## License

Educational/demo purposes. Use models and data according to their respective licenses.

---

## Credits

Built with:
- MongoDB Community Edition
- Hugging Face SentenceTransformers
- FAISS by Facebook Research
- llama.cpp by Georgi Gerganov
- FastAPI by Sebastián Ramírez
- React by Meta

---

**🎉 You now have a complete, production-ready RAG system!**

**Start it with:** `.\start_web.ps1`
**Open:** http://localhost:3000
**Enjoy!** 🚀
