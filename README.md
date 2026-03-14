# PDF RAG with Gemini Embedding

A Python program that ingests a PDF file (including pages with **text**, **tables**, and **images**) and builds a Retrieval-Augmented Generation (RAG) pipeline powered by **Gemini Embedding** (`text-embedding-004` — Gemini Embedding 2).

---

## How it works

```
PDF file
   │
   ├─ pdfplumber  → text chunks + table chunks (markdown rows)
   └─ PyMuPDF     → images → Gemini Vision → text descriptions
          │
          ▼
   Gemini text-embedding-004  (embeds every chunk)
          │
          ▼
   In-memory vector store  (cosine similarity search)
          │
   User question → embed → retrieve top-k chunks → Gemini Flash → answer
```

### Components

| Component | Library / Model |
|-----------|----------------|
| Text & table extraction | `pdfplumber` |
| Image extraction | `PyMuPDF (fitz)` |
| Image description | `gemini-2.0-flash` (vision) |
| Embedding | `text-embedding-004` (Gemini Embedding 2) |
| Generation | `gemini-2.0-flash` |
| Vector store | NumPy cosine similarity (in-memory, JSON-serialisable) |

---

## Prerequisites

1. **Python 3.9+**
2. A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Set the API key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

### 1 — Ingest a PDF and save the index

```bash
python pdf_rag.py ingest path/to/document.pdf --index document.json
```

This command:
- Extracts all text, tables, and images from the PDF.
- Generates text descriptions for each image using Gemini Vision.
- Embeds every chunk with `text-embedding-004`.
- Saves the index to `document.json` for later reuse.

---

### 2 — Query a saved index

```bash
python pdf_rag.py query "What are the key findings?" --index document.json
```

Optional: `--top-k 5` controls how many chunks are retrieved (default 5).

---

### 3 — One-shot: ingest + answer in one command

```bash
python pdf_rag.py ask path/to/document.pdf "Summarise the tables in this report"
```

Add `--index document.json` to also persist the index.

---

## Example session

```
$ python pdf_rag.py ingest annual_report.pdf --index annual_report.json
[1/3] Extracting content from 'annual_report.pdf' …
      Found 47 raw chunks.
[2/3] Describing images and embedding all chunks …
      Describing image on page 3, image 1 …
      Describing image on page 7, image 1 …
      Embedded 47 chunks.
[3/3] Ingestion complete.
  [info] Index saved to annual_report.json

$ python pdf_rag.py query "What was the revenue in 2023?" --index annual_report.json
  [info] Index loaded from annual_report.json (47 chunks)

=== Answer ===
According to the financial table on page 4, the total revenue for 2023 was $4.2 billion,
representing a 12% increase compared to the prior year.
```

---

## Configuration

Constants at the top of `pdf_rag.py` can be adjusted:

| Constant | Default | Description |
|----------|---------|-------------|
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Embedding model name |
| `GEMINI_GENERATION_MODEL` | `gemini-2.0-flash` | Generation model name |
| `CHUNK_SIZE` | `600` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Retrieved chunks per query |

---

## Notes

- The vector index is stored as plain JSON, so it is portable and human-readable.
- Images are sent to Gemini Vision to produce textual descriptions before embedding; the raw image bytes are **not** persisted in the index.
- For very large PDFs, consider batching the embedding calls or using a persistent vector database such as ChromaDB or Pinecone.
