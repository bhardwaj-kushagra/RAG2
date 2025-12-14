import React, { useState, useEffect } from 'react';
import './App.css';
import { apiService } from './apiService';

function App() {
  const [activeTab, setActiveTab] = useState('ingest');
  const [apiStatus, setApiStatus] = useState('checking');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Query state
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(3);
  const [agentGoal, setAgentGoal] = useState('');
  const [agentTopK, setAgentTopK] = useState(5);

  // Check API health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      await apiService.checkHealth();
      setApiStatus('online');
    } catch (err) {
      setApiStatus('offline');
    }
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    setResult(null);
    setError(null);
  };

  const handleIngest = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one file to ingest');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiService.ingestFiles(selectedFiles);
      setResult(response);
      setSelectedFiles([]);
    } catch (err) {
      setError(`Failed to ingest files: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleBuildIndex = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiService.buildIndex();
      setResult(response);
    } catch (err) {
      setError(`Failed to build index: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiService.query(question, topK);
      setResult(response);
    } catch (err) {
      setError(`Query failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrieve = async () => {
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiService.retrieve(question, topK);
      setResult(response);
    } catch (err) {
      setError(`Retrieval failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const resetResults = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <div className="header">
        <h1>🚀 Local RAG System</h1>
        <p>Retrieval-Augmented Generation with MongoDB, FAISS & Llama</p>
      </div>

      <div className="status-bar">
        <div className="status-indicator">
          <div className={`status-dot ${apiStatus === 'online' ? '' : 'offline'}`}></div>
          <span>API Server: {apiStatus === 'online' ? '✓ Online' : '✗ Offline'}</span>
        </div>
        <button className="button button-secondary" onClick={checkHealth}>
          Refresh Status
        </button>
      </div>

      <div className="tabs">
        <div className="tab-buttons">
          <button
            className={`tab-button ${activeTab === 'ingest' ? 'active' : ''}`}
            onClick={() => { setActiveTab('ingest'); resetResults(); }}
          >
            📁 Ingest Files
          </button>
          <button
            className={`tab-button ${activeTab === 'index' ? 'active' : ''}`}
            onClick={() => { setActiveTab('index'); resetResults(); }}
          >
            🔍 Build Index
          </button>
          <button
            className={`tab-button ${activeTab === 'query' ? 'active' : ''}`}
            onClick={() => { setActiveTab('query'); resetResults(); }}
          >
            💬 Query RAG
          </button>
          <button
            className={`tab-button ${activeTab === 'retrieve' ? 'active' : ''}`}
            onClick={() => { setActiveTab('retrieve'); resetResults(); }}
          >
            📚 Retrieve Only
          </button>
          <button
            className={`tab-button ${activeTab === 'agent' ? 'active' : ''}`}
            onClick={() => { setActiveTab('agent'); resetResults(); }}
          >
            🧠 Agentic AI
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'ingest' && (
            <div>
              <div className="section">
                <h3>Upload Documents</h3>
                <p>
                  Select one or more files to ingest into MongoDB. Supported formats: TXT, PDF, DOCX, CSV, XLSX, JSON, XML, PNG/JPG (OCR)
                </p>
                
                <div className="file-upload">
                  <input
                    type="file"
                    id="file-input"
                    multiple
                    onChange={handleFileSelect}
                    accept=".txt,.pdf,.docx,.csv,.xlsx,.json,.xml,.png,.jpg,.jpeg,.tif,.tiff"
                  />
                  <label htmlFor="file-input">
                    <div>
                      <h3 style={{ marginBottom: '10px' }}>📤 Click to Upload Files</h3>
                      <p>or drag and drop</p>
                    </div>
                  </label>
                </div>

                {selectedFiles.length > 0 && (
                  <div className="file-list">
                    <h4>Selected Files ({selectedFiles.length}):</h4>
                    {selectedFiles.map((file, idx) => (
                      <div key={idx} className="file-item">
                        <span>{file.name}</span>
                        <span style={{ color: '#666', fontSize: '0.9em' }}>
                          {(file.size / 1024).toFixed(1)} KB
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                <div style={{ marginTop: '20px' }}>
                  <button
                    className="button"
                    onClick={handleIngest}
                    disabled={loading || selectedFiles.length === 0}
                  >
                    {loading ? 'Ingesting...' : 'Ingest Files'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'index' && (
            <div>
              <div className="section">
                <h3>Build FAISS Index</h3>
                <p>
                  Build a vector index from all passages stored in MongoDB. This creates embeddings using your local llama.cpp embedding model (with a SentenceTransformers fallback) and builds a FAISS index for fast similarity search.
                </p>

                <div className="info-box">
                  <strong>ℹ️ Note:</strong> Make sure you've ingested documents first. The index will be saved to <code>faiss.index</code> and <code>id_map.json</code> in your project directory.
                </div>

                <button
                  className="button"
                  onClick={handleBuildIndex}
                  disabled={loading}
                >
                  {loading ? 'Building Index...' : 'Build Index'}
                </button>
              </div>
            </div>
          )}

          {activeTab === 'query' && (
            <div>
              <div className="section">
                <h3>Ask a Question</h3>
                <p>
                  Query the RAG system with a question. The system will retrieve relevant passages and generate an answer using your local Llama model.
                </p>

                <div className="input-group">
                  <label htmlFor="question">Your Question:</label>
                  <textarea
                    id="question"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="What is machine learning?"
                    disabled={loading}
                  />
                </div>

                <div className="input-group">
                  <label htmlFor="topk">Number of passages to retrieve (k):</label>
                  <input
                    id="topk"
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    min="1"
                    max="10"
                    disabled={loading}
                  />
                </div>

                <button
                  className="button"
                  onClick={handleQuery}
                  disabled={loading || !question.trim()}
                >
                  {loading ? 'Processing...' : 'Generate Answer'}
                </button>
              </div>
            </div>
          )}

          {activeTab === 'retrieve' && (
            <div>
              <div className="section">
                <h3>Retrieve Passages Only</h3>
                <p>
                  Search for relevant passages without generating an answer. Useful for testing retrieval quality.
                </p>

                <div className="input-group">
                  <label htmlFor="retrieve-question">Your Question:</label>
                  <textarea
                    id="retrieve-question"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="What is neural networks?"
                    disabled={loading}
                  />
                </div>

                <div className="input-group">
                  <label htmlFor="retrieve-topk">Number of passages to retrieve (k):</label>
                  <input
                    id="retrieve-topk"
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    min="1"
                    max="10"
                    disabled={loading}
                  />
                </div>

                <button
                  className="button"
                  onClick={handleRetrieve}
                  disabled={loading || !question.trim()}
                >
                  {loading ? 'Retrieving...' : 'Retrieve Passages'}
                </button>
              </div>
            </div>
          )}

          {activeTab === 'agent' && (
            <div>
              <div className="section">
                <h3>Agentic AI (Goal-driven)</h3>
                <p>
                  Give the agent a high-level goal. It can run a quick RAG evaluation,
                  create a note, inspect the database, or search and summarize using
                  the existing RAG pipeline.
                </p>

                <div className="input-group">
                  <label htmlFor="agent-goal">Agent Goal:</label>
                  <textarea
                    id="agent-goal"
                    value={agentGoal}
                    onChange={(e) => setAgentGoal(e.target.value)}
                    placeholder="Examples: \n- Run a quick evaluation of our RAG quality.\n- Summarize what we know about HelioScope.\n- Inspect what is stored in rag_db.passages."
                    disabled={loading}
                  />
                </div>

                <div className="input-group">
                  <label htmlFor="agent-topk">Passages to retrieve for search/summarize (k):</label>
                  <input
                    id="agent-topk"
                    type="number"
                    value={agentTopK}
                    onChange={(e) => setAgentTopK(parseInt(e.target.value) || 1)}
                    min="1"
                    max="10"
                    disabled={loading}
                  />
                </div>

                <button
                  className="button"
                  onClick={async () => {
                    if (!agentGoal.trim()) {
                      setError('Please enter an agent goal');
                      return;
                    }
                    setLoading(true);
                    setError(null);
                    setResult(null);
                    try {
                      const response = await apiService.runAgent(agentGoal, agentTopK);
                      setResult({ agent: response });
                    } catch (err) {
                      setError(`Agent request failed: ${err.message}`);
                    } finally {
                      setLoading(false);
                    }
                  }}
                  disabled={loading || !agentGoal.trim()}
                >
                  {loading ? 'Running Agent...' : 'Run Agent'}
                </button>
              </div>
            </div>
          )}

          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <div>Processing your request...</div>
            </div>
          )}

          {error && (
            <div className="result-box error">
              <h4>❌ Error</h4>
              <p>{error}</p>
            </div>
          )}

          {result && !error && (
            <div className="result-box success">
              {result.answer && (
                <>
                  <h4>✨ Generated Answer</h4>
                  <div className="answer-text">{result.answer}</div>
                </>
              )}

              {result.agent && (
                <>
                  <h4>🧠 Agent Result</h4>
                  <p><strong>Mode:</strong> {result.agent.mode}</p>
                  <div className="answer-text">{result.agent.answer}</div>
                </>
              )}

              {result.passages && result.passages.length > 0 && (
                <div className="passages">
                  <h4>📚 Retrieved Passages ({result.passages.length})</h4>
                  {result.passages.map((passage, idx) => (
                    <div key={idx} className="passage-item">
                      <div className="passage-source">Source: {passage.source_id}</div>
                      <div className="passage-text">{passage.text}</div>
                    </div>
                  ))}
                </div>
              )}

              {result.message && !result.passages && (
                <>
                  <h4>✅ Success</h4>
                  <p>{result.message}</p>
                  {result.passages_ingested !== undefined && (
                    <p><strong>Passages ingested:</strong> {result.passages_ingested}</p>
                  )}
                  {result.index_size !== undefined && (
                    <p><strong>Index size:</strong> {result.index_size} vectors</p>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
