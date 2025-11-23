# Ingestion Module Guide

This document explains how to use the extended ingestion module to load content into the RAG system from multiple sources and formats.

## Supported Inputs

File formats (placed in `data/docs/`):
- Text: `.txt`
- PDF: `.pdf` (embedded text; OCR fallback if no text and pdf2image + pytesseract are available)
- Word: `.docx`
- Tabular: `.csv`, `.xlsx` (rows concatenated with `|` separators)
- JSON: `.json` (recursive extraction of primitive values)
- XML: `.xml` (all text nodes concatenated)
- Images (OCR): `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`

External sources:
- Oracle DB rows (query returning ID, TITLE, BODY)
- Another MongoDB collection (documents with a text field)

All ingested text is chunked (250 words, 50 overlap) and stored in MongoDB `rag_db.passages` with fields:
```
{ source, source_id, text, created_at }
```

## Optional Dependencies
Install only what you need:
```powershell
pip install PyPDF2 python-docx pandas pytesseract pdf2image oracledb pillow
```

External tools (for OCR & scanned PDFs):
- Tesseract OCR (add install path to PATH)
- Poppler (for pdf2image on Windows; add Poppler `bin` directory to PATH)

## CLI Flags
```
--ingest-files                 Ingest all supported files under data/docs/
--ingest-oracle                Ingest from Oracle DB (requires DSN, user, password)
--oracle-dsn <dsn>             Oracle DSN (host:port/service)
--oracle-user <user>           Oracle username
--oracle-password <pass>       Oracle password
--oracle-query <sql>           SQL query returning ID, TITLE, BODY (default SELECT ID, TITLE, BODY FROM DOCUMENTS)
--ingest-mongo-source          Ingest from another Mongo collection
--src-mongo-uri <uri>          Source Mongo URI
--src-mongo-db <db>            Source Mongo DB name
--src-mongo-coll <coll>        Source Mongo collection name
--src-mongo-text-field <f>     Field containing text (default text)
--src-mongo-id-field <f>       Optional field used for source_id (otherwise _id)
```

## Examples

### 1. Ingest All Files
```powershell
python src/rag_windows.py --ingest-files
```

### 2. Ingest From Oracle
```powershell
python src/rag_windows.py --ingest-oracle --oracle-dsn "host:port/service" `
  --oracle-user USER --oracle-password PASS `
  --oracle-query "SELECT ID, TITLE, BODY FROM DOCUMENTS" 
```

### 3. Ingest From Another MongoDB Collection
```powershell
python src/rag_windows.py --ingest-mongo-source `
  --src-mongo-uri "mongodb://localhost:27017" `
  --src-mongo-db other_db --src-mongo-coll articles `
  --src-mongo-text-field content --src-mongo-id-field slug
```

### 4. Build Index & Query
```powershell
python src/rag_windows.py --build-index
python src/rag_windows.py --query "What is RAG?" --k 5
```

## How OCR Works
1. For PDFs: attempt embedded text extraction via PyPDF2.
2. If no text returned and `pdf2image` + `pytesseract` + `Pillow` present, rasterize pages and run OCR.
3. For image files: direct OCR via `pytesseract.image_to_string`.
4. Missing OCR dependencies ⇒ file skipped (warning logged).

## Design Extensibility
The ingestion logic uses an `EXTRACTORS` mapping from file extension to extraction function. Adding a new format is as simple as writing a function that takes a `Path` and returns a `str`, then registering its extension.

Example adding Markdown:
```python
def _extract_md(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")
EXTRACTORS[".md"] = _extract_md
SUPPORTED_FILE_EXT.add(".md")
```

## Oracle Table Requirements
Query should return at least three columns usable as `(id, title, body)`. Adjust unpacking if your schema differs.

## Source Mongo Requirements
Each source document must have a text field (`--src-mongo-text-field`). Optional ID field used for semantic `source_id`; fallback `_id` is used if not provided.

## Troubleshooting
- Missing dependency warnings: install the optional packages listed above.
- Large PDFs slow: consider limiting pages or pre-processing externally.
- OCR quality poor: install language data for Tesseract (e.g., `eng`), ensure image resolution is adequate.
- Oracle errors: verify DSN, firewall, and that `oracledb` is properly installed.
- Memory issues building index: reduce number of documents per ingestion or use a smaller embedding model.

## Recommended Installation Bundle
Minimal (files only):
```powershell
pip install PyPDF2 python-docx pandas
```
With OCR & Oracle:
```powershell
pip install PyPDF2 python-docx pandas pytesseract pdf2image oracledb pillow
```

## Next Steps
- Add `--chunk-size` and `--chunk-overlap` CLI controls
- Add metadata filters (e.g., by source type) during retrieval
- Implement reranking or hybrid (BM25 + dense) retrieval
- Persist embeddings separately for incremental updates

## License Reminder
Ensure compliance with licenses for models, libraries, and any proprietary data ingested.
