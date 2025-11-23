const API_BASE_URL = 'http://localhost:8000';

export const apiService = {
  async checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },

  async ingestFiles(files) {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    
    const response = await fetch(`${API_BASE_URL}/ingest/files`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  },

  async buildIndex(embeddingModel = 'all-MiniLM-L6-v2') {
    const response = await fetch(`${API_BASE_URL}/build-index`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ embedding_model: embeddingModel }),
    });
    return response.json();
  },

  async query(question, k = 3) {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question, k }),
    });
    return response.json();
  },

  async retrieve(question, k = 3) {
    const response = await fetch(`${API_BASE_URL}/retrieve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question, k }),
    });
    return response.json();
  },
};
