# RAG System - Web UI

React-based web interface for the Local RAG System.

## Quick Start

### Prerequisites
- Node.js 16+ installed
- RAG backend API server running on port 8000
- MongoDB running locally

### Development

```bash
# Install dependencies (first time only)
npm install

# Start development server
npm start
```

The app will open at `http://localhost:3000` with hot-reload enabled.

### Production Build

```bash
# Create optimized production build
npm run build

# The build/ folder contains the static files
```

## Features

- **File Upload**: Drag-and-drop or browse to upload documents (TXT, PDF, DOCX, CSV, XLSX, JSON, XML, images)
- **Index Management**: Build FAISS index from ingested documents
- **RAG Queries**: Ask questions and get AI-generated answers with source citations
- **Retrieve Only**: Test retrieval without LLM generation
- **Status Monitoring**: Real-time API server health check

## Project Structure

```
web/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── App.js              # Main component with tabs
│   ├── App.css             # Styling
│   ├── apiService.js       # API client
│   ├── index.js            # React entry point
│   └── index.css           # Global styles
├── package.json            # Dependencies and scripts
└── README.md               # This file
```

## Configuration

### Change API URL

Edit `src/apiService.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';  // Change port if needed
```

Also update `package.json` proxy:

```json
"proxy": "http://localhost:8000"
```

## API Backend

Make sure the FastAPI backend is running:

```powershell
# From project root
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
```

The backend must have CORS enabled for `http://localhost:3000` (already configured in `src/api.py`).

## Troubleshooting

### Port 3000 Already in Use

```bash
# Windows PowerShell
$env:PORT=3001; npm start

# Linux/Mac
PORT=3001 npm start
```

### Cannot Connect to API

1. Verify API is running: `http://localhost:8000/health`
2. Check browser console for CORS errors
3. Ensure API CORS middleware includes `http://localhost:3000`

### npm install Fails

```bash
# Clear cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

## Documentation

See `../docs/WEB_UI_GUIDE.md` for complete usage instructions and deployment guide.

## Technologies

- **React 18**: UI framework
- **React Scripts 5**: Build tooling
- **Fetch API**: HTTP client (no external dependencies like Axios)
