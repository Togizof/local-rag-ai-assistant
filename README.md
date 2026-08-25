# Local RAG AI Assistant

A simple offline AI assistant that answers questions based on your local documents. This project runs completely on your local machine using **Microsoft Foundry Local** for model inference and **SQLite** to store text chunks and vector embeddings. No internet connection or API keys are required.

---

## Features

- **100% Offline:** All data stays on your local machine. No data is sent to the cloud.
- **Local Embeddings:** Uses the `qwen3-embedding-0.6b` model to create text vectors.
- **Local LLM:** Uses the `phi-3.5-mini` model to generate responses based on your files.
- **Smart Chunking:** Splits documents into chunks while respecting sentence boundaries.
- **Semantic Search:** Uses Cosine Similarity to retrieve the most relevant document passages.
- **Sources Citing:** The assistant references the exact document it used to answer the question.
- **Two Interfaces:** Includes a Streamlit Web Dashboard and a Command Line Interface (CLI).

---

## Project Structure

- `app.py`: Streamlit web application.
- `cli.py`: Interactive command-line chat interface.
- `ingest.py`: Script to process documents and save vectors into SQLite.
- `test_rag.py`: Unit tests for similarity calculations and SQLite database operations.
- `src/`: Core Python modules containing database, embedding, LLM, and pipeline logic.
- `data/docs/`: Directory where you place your input text/markdown files.

---

## Installation & Setup

### 1. Clone the repository
Navigate to your local project directory.

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the virtual environment
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Place your documents
Put your `.txt` or `.md` files in the `data/docs/` directory.

### Step 2: Index your documents
Process and save your documents to the SQLite database:
```bash
python ingest.py --clear
```
*(Note: On the first run, the local models will download automatically. Subsequent runs will be fully offline).*

### Step 3: Run the application

#### Option A: Streamlit Web UI (Recommended)
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

#### Option B: Console Chat (CLI)
```bash
python cli.py
```

---

## Running Tests

To verify that the vector database and search math work correctly, run the unit test suite:
```bash
python -m unittest test_rag.py
```
