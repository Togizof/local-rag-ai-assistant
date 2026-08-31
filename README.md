# My Library Assistant 📚

This is a local, offline AI assistant designed to help you search and query your reading notes and book summaries. It runs completely offline using **Microsoft Foundry Local** for model inference and **SQLite** to store text chunks and vector embeddings.

It supports bilingual Q&A, allowing you to ask questions in Turkish even if your reading notes are in English.

---

## Features

- **100% Offline:** All reading notes stay private on your local machine. No internet or external API keys needed.
- **Bilingual Q&A:** Ask questions in Turkish, and the assistant will search and translate English book notes to answer you in Turkish.
- **SQLite Database:** Stores text passages and vectors locally.
- **Streamlit Web UI:** A clean, book-themed dashboard to upload notes and chat with your library assistant.
- **Unit Tests:** Simple tests to verify vector math and database operations.

---

## Notion Integration (How to Import Your Notes)

If you keep your reading notes in Notion, you can import them into this assistant in a few easy steps:

1. **Open Notion:** Go to the page or database containing your reading notes.
2. **Open Export Menu:** Click the three dots `...` in the top right corner of the Notion page.
3. **Select Export:** Click **Export**.
4. **Choose Format:** Set the **Export format** to `Markdown & CSV`.
5. **Download:** Click **Export**. Notion will download a ZIP file containing your pages as Markdown (`.md`) files.
6. **Extract:** Extract the ZIP file.
7. **Copy Files:** Copy the extracted `.md` files and paste them into the `data/docs/` directory of this project.
8. **Index:** Run the indexing script to build your local library database:
   ```bash
   python ingest.py --clear
   ```
Now you can search all your Notion notes offline!

---

## Installation & Setup

### 1. Create a virtual environment
```bash
python -m venv .venv
```

### 2. Activate the virtual environment
- **Windows:**
  ```powershell
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Place your book notes
Put your `.txt` or `.md` files (exported from Notion or elsewhere) in the `data/docs/` directory.

### Step 2: Index documents
Build your local database:
```bash
python ingest.py --clear
```

### Step 3: Start the application
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Running Tests

To run the unit tests:
```bash
python -m unittest test_rag.py
```
