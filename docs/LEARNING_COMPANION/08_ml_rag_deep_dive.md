# 🧠 Deep Dive: The Machine Learning & RAG Pipeline

This document is a technical script and explanation guide for the "Machine Learning" portion of the demonstration. It explains exactly what happens to data from the moment it enters the system to the moment an answer is generated.

---

## 1. The High-Level Pipeline

**Concept:** RAG (Retrieval-Augmented Generation) is a hack to give a "frozen" AI model access to "fresh" or "private" data.

**The Flow:**
1.  **Ingest:** Read raw files (PDF, CSV, etc.).
2.  **Chunk:** Break text into small, digestible pieces.
3.  **Embed:** Convert text pieces into mathematical vectors (lists of numbers).
4.  **Index:** Store vectors in a searchable structure (FAISS).
5.  **Retrieve:** Find the most relevant vectors for a user's question.
6.  **Generate:** Feed the relevant text + question to the LLM to write an answer.

---

## 2. Step-by-Step Technical Breakdown

### Step A: Ingestion & Chunking (The "Pre-processing")

**The Problem:** LLMs have a limit on how much text they can read at once (Context Window). We can't feed a 100-page PDF into a small model.

**The Solution:** Chunking.

**How it works in our code (`src/rag_windows.py`):**
*   **Tool:** Custom Python logic (or libraries like `langchain` text splitters).
*   **Process:**
    1.  Extract raw text from file (using `pypdf`, `pandas`, etc.).
    2.  Split text into blocks of **250 words**.
    3.  **Overlap:** We include the last **50 words** of Chunk A at the start of Chunk B.
        *   *Why?* If a sentence like "The password is... [CUT] ...12345" gets split, the meaning is lost. Overlap preserves context across boundaries.

**Visual Analogy:** It's like cutting a long movie film into 30-second clips so you can find the exact scene later.

---

### Step B: Tokenization (The "Translation")

**The Problem:** Computers don't understand English characters. They only understand numbers.

**The Solution:** Tokenization.

**How it works:**
*   **Tool:** `SentenceTransformers` (using the `all-MiniLM-L6-v2` tokenizer).
*   **Process:**
    *   Input: "Hello World"
    *   The tokenizer looks up a dictionary.
    *   "Hello" -> ID `7592`
    *   "World" -> ID `2088`
    *   Output: `[7592, 2088]`
*   **Note:** A "token" is roughly 0.75 of a word. "Ingestion" might be two tokens: "Ingest" + "ion".

---

### Step C: Embedding (The "Meaning")

**The Problem:** We have numbers (tokens), but we need *meaning*. We need to know that "Dog" is similar to "Puppy" but different from "Car".

**The Solution:** Vector Embeddings.

**How it works:**
*   **Tool:** `SentenceTransformers` model (`all-MiniLM-L6-v2`).
*   **Architecture:** This is a **BERT-based** neural network (Transformer).
*   **Process:**
    1.  The model reads the sequence of tokens.
    2.  It passes them through 6 layers of "Self-Attention".
    3.  **Self-Attention:** The model calculates how much every word relates to every other word in the sentence.
    4.  **Pooling:** The model outputs a vector for every token. We take the **average** (Mean Pooling) of all these vectors to get one single vector for the whole sentence.
    5.  **Normalization:** We scale the vector so its length is exactly 1.0 (L2 Normalization).

**The Output:**
A list of **384 floating-point numbers**.
`[-0.023, 0.154, 0.871, ...]`

**What this means:**
Imagine a 3D graph (X, Y, Z). A point is a location.
Now imagine a **384-D graph**.
*   The vector for "King" is at location A.
*   The vector for "Queen" is at location B.
*   The vector for "Apple" is at location C.
*   Mathematically, the distance between A and B is very small. The distance between A and C is very large.
*   **We have turned meaning into geometry.**

---

### Step D: Indexing (The "Database")

**The Problem:** We have 10,000 vectors. To find the closest one to a query, we'd have to compare the query to *all* 10,000. That's slow.

**The Solution:** FAISS (Facebook AI Similarity Search).

**How it works:**
*   **Tool:** `faiss-cpu`.
*   **Index Type:** `IndexFlatIP` (Inner Product).
*   **Process:**
    *   Since our vectors are normalized (length = 1), the "Inner Product" (Dot Product) is identical to **Cosine Similarity**.
    *   Cosine Similarity measures the angle between two vectors.
    *   0 degrees = Identical meaning (Score 1.0).
    *   90 degrees = Unrelated (Score 0.0).
    *   180 degrees = Opposite meaning (Score -1.0).

**In our code:**
We save this structure to `faiss.index`. It's a highly optimized binary file designed for super-fast math operations.

---

### Step E: Retrieval (The "Search")

**The User Query:** "Who is the ML Researcher?"

**The Process:**
1.  **Embed Query:** Convert the question into a 384-D vector using the *same* model as Step C.
2.  **Search:** Send this vector to FAISS.
3.  **Math:** FAISS calculates the dot product between the Query Vector and all Document Vectors.
4.  **Rank:** It sorts them by score (highest to lowest).
5.  **Select:** We take the `top_k` (e.g., top 3) results.
6.  **Fetch:** We use the IDs from FAISS to look up the actual text in MongoDB.

---

### Step F: Generation (The "Intelligence")

**The Problem:** We have the relevant text chunks, but the user wants a natural language answer, not a list of paragraphs.

**The Solution:** Large Language Model (LLM).

**The Model:**
*   **Name:** `TinyLlama-1.1B-Chat` (or similar).
*   **Format:** **GGUF** (GPT-Generated Unified Format).
*   **Quantization:** `Q4_K_M`.
    *   Standard models use 16-bit numbers (float16) for weights.
    *   **Q4** means we compress those weights to **4-bit integers**.
    *   **Result:** The model is 4x smaller and runs 4x faster, with very little loss in intelligence. This allows it to run on your CPU RAM instead of a massive GPU.

**The Prompt Engineering:**
We don't just send the question. We construct a specific "Prompt Template":

```text
<|system|>
You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not in the context, say "I don't know".

Context:
[Chunk 1 text...]
[Chunk 2 text...]
</s>
<|user|>
Who is the ML Researcher?
</s>
<|assistant|>
```

**The Inference Loop:**
1.  **Input:** The model receives this huge block of text (System Prompt + Context + Question).
2.  **Prediction:** It calculates the probability of the *next single token*.
    *   "Based on this text, the most likely next word is 'Bob'."
3.  **Output:** It outputs "Bob".
4.  **Loop:** It adds "Bob" to the input and predicts the *next* token.
    *   "Based on '...is Bob', the next word is 'Smith'."
5.  **Stop:** It continues until it predicts a special "End of Sentence" token (`</s>`).

---

## 3. Summary of Tools Used

| Component | Tool/Library | Role |
| :--- | :--- | :--- |
| **Orchestrator** | Python (FastAPI) | Controls the flow of data. |
| **Database** | MongoDB | Stores the raw text and metadata. |
| **Embedder** | SentenceTransformers | Converts text to numbers (Vectors). |
| **Vector DB** | FAISS | Finds similar vectors fast. |
| **LLM Runtime** | llama-cpp-python | Runs the AI model on CPU. |
| **Model Format** | GGUF | Optimized file format for local inference. |

---

## 4. Why this is "Secure" & "Private"

1.  **No API Calls:** At no point does data leave the machine. The embedding model runs locally. The LLM runs locally.
2.  **Air-Gapped Capable:** You could unplug the internet, and this system would still work perfectly (once models are downloaded).
3.  **Data Sovereignty:** You own the database (MongoDB) and the Index (FAISS). No third party sees your documents.
