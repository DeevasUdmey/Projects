"""
PDF RAG Pipeline using Gemini Embedding (text-embedding-004).

Extracts text, tables, and images from a PDF, embeds all content using
Gemini Embeddings, and answers questions through retrieval-augmented generation.

Usage:
    # Ingest a PDF and save the index
    python pdf_rag.py ingest path/to/file.pdf --index my_index.json

    # Query against a saved index
    python pdf_rag.py query "What is this document about?" --index my_index.json

    # One-shot: ingest and immediately query
    python pdf_rag.py ask path/to/file.pdf "What is this document about?"

Requirements:
    Set the GEMINI_API_KEY environment variable before running.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports — provide helpful messages if missing
# ---------------------------------------------------------------------------
try:
    import pdfplumber
except ImportError:
    sys.exit("Missing dependency: pip install pdfplumber")

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Missing dependency: pip install pymupdf")

try:
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency: pip install Pillow")

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("Missing dependency: pip install google-genai")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_EMBEDDING_MODEL = "text-embedding-004"  # Gemini Embedding 2
GEMINI_GENERATION_MODEL = "gemini-2.0-flash"
CHUNK_SIZE = 600          # characters per text chunk
CHUNK_OVERLAP = 100       # character overlap between chunks
TOP_K = 5                 # retrieved chunks per query
MIN_CHUNK_LEN = 50        # discard very short fragments
IMAGE_DESCRIPTION_PREFIX = "[IMAGE DESCRIPTION] "
TABLE_PREFIX = "[TABLE] "


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping fixed-size chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if len(c.strip()) >= MIN_CHUNK_LEN]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64-encoded string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# PDF Processor
# ---------------------------------------------------------------------------

class PDFProcessor:
    """Extracts text, tables, and images from a PDF file."""

    def extract_content(self, pdf_path: str) -> List[Dict]:
        """
        Return a list of content chunks, each being a dict:
          {
            "type": "text" | "table" | "image_description",
            "content": <str>,
            "page": <int>,
            "source": <str>,   # human-readable label
          }
        """
        chunks: List[Dict] = []
        chunks.extend(self._extract_text_and_tables(pdf_path))
        chunks.extend(self._extract_images(pdf_path))
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_text_and_tables(self, pdf_path: str) -> List[Dict]:
        chunks: List[Dict] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # --- tables (extract before text so we can remove them) ---
                tables = page.extract_tables()
                for tbl_idx, table in enumerate(tables):
                    table_text = self._table_to_text(table)
                    if len(table_text.strip()) >= MIN_CHUNK_LEN:
                        chunks.append(
                            {
                                "type": "table",
                                "content": TABLE_PREFIX + table_text,
                                "page": page_num,
                                "source": f"page {page_num}, table {tbl_idx + 1}",
                            }
                        )

                # --- plain text (strip table bounding-boxes first) ---
                table_bboxes = [tbl.bbox for tbl in page.find_tables()]
                text_page = page
                for bbox in table_bboxes:
                    text_page = text_page.filter(
                        lambda obj, bb=bbox: not (
                            bb[0] <= obj["x0"] and obj["x1"] <= bb[2]
                            and bb[1] <= obj["top"] and obj["bottom"] <= bb[3]
                        )
                    )

                raw_text = text_page.extract_text() or ""
                raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

                for chunk in _chunk_text(raw_text):
                    chunks.append(
                        {
                            "type": "text",
                            "content": chunk,
                            "page": page_num,
                            "source": f"page {page_num}",
                        }
                    )

        return chunks

    @staticmethod
    def _table_to_text(table: List[List[Optional[str]]]) -> str:
        """Convert a pdfplumber table (list-of-lists) to a markdown-style string."""
        rows = []
        for row in table:
            cells = [str(c).strip() if c is not None else "" for c in row]
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _extract_images(self, pdf_path: str) -> List[Dict]:
        """Use PyMuPDF to extract embedded images from the PDF."""
        chunks: List[Dict] = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    b64 = _pil_to_base64(image)
                    chunks.append(
                        {
                            "type": "image",
                            "content": b64,          # raw base64 for Gemini vision
                            "page": page_num + 1,
                            "source": f"page {page_num + 1}, image {img_idx + 1}",
                        }
                    )
                except Exception as exc:
                    print(
                        f"  [warning] Could not extract image on page {page_num + 1}: {exc}",
                        file=sys.stderr,
                    )
        doc.close()
        return chunks


# ---------------------------------------------------------------------------
# Gemini client wrapper
# ---------------------------------------------------------------------------

class GeminiClient:
    """Thin wrapper around google.genai for embeddings and generation."""

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)
        self._embed_model = GEMINI_EMBEDDING_MODEL
        self._gen_model = GEMINI_GENERATION_MODEL

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Return a single embedding vector for *text*."""
        response = self._client.models.embed_content(
            model=self._embed_model,
            contents=[text],
            config=genai_types.EmbedContentConfig(task_type=task_type),
        )
        return response.embeddings[0].values

    def describe_image(self, b64_image: str) -> str:
        """Ask Gemini to produce a textual description of a base64-encoded image."""
        image_bytes = base64.b64decode(b64_image)
        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        response = self._client.models.generate_content(
            model=self._gen_model,
            contents=[
                image_part,
                (
                    "Describe this image in detail, including any text, charts, "
                    "diagrams, or visual information it contains."
                ),
            ],
        )
        return response.text.strip()

    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer to *question* using *context* as grounding."""
        prompt = textwrap.dedent(
            f"""
            You are a helpful assistant. Use ONLY the context below to answer
            the question. If the answer is not in the context, say so clearly.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
            """
        ).strip()
        response = self._client.models.generate_content(
            model=self._gen_model,
            contents=[prompt],
        )
        return response.text.strip()


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Simple in-memory vector store backed by numpy cosine similarity.
    Can be serialised to / deserialised from a JSON file.
    """

    def __init__(self):
        self._embeddings: List[List[float]] = []
        self._metadata: List[Dict] = []

    # ------------------------------------------------------------------
    def add(self, embedding: List[float], metadata: Dict) -> None:
        self._embeddings.append(embedding)
        self._metadata.append(metadata)

    def search(self, query_embedding: List[float], top_k: int = TOP_K) -> List[Tuple[float, Dict]]:
        """Return top-k (score, metadata) pairs sorted by descending similarity."""
        if not self._embeddings:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        matrix = np.array(self._embeddings, dtype=np.float32)
        scores = matrix @ q / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self._metadata[i]) for i in top_indices]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the vector store to a JSON file."""
        data = {
            "embeddings": self._embeddings,
            "metadata": self._metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  [info] Index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a previously saved vector store from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store = cls()
        store._embeddings = data["embeddings"]
        store._metadata = data["metadata"]
        print(f"  [info] Index loaded from {path} ({len(store._embeddings)} chunks)")
        return store

    def __len__(self) -> int:
        return len(self._embeddings)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Extract content from a PDF (text, tables, images).
      2. Embed each chunk with Gemini Embedding (text-embedding-004).
      3. Store embeddings in an in-memory vector store.
      4. At query time, embed the question and retrieve the most relevant chunks.
      5. Feed the retrieved context to Gemini to generate a grounded answer.
    """

    def __init__(self, api_key: str):
        self._client = GeminiClient(api_key)
        self._processor = PDFProcessor()
        self._store = VectorStore()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, pdf_path: str) -> None:
        """
        Extract, describe, embed, and index all content from *pdf_path*.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"[1/3] Extracting content from '{path.name}' …")
        raw_chunks = self._processor.extract_content(pdf_path)
        print(f"      Found {len(raw_chunks)} raw chunks.")

        print("[2/3] Describing images and embedding all chunks …")
        embedded = 0
        for chunk in raw_chunks:
            if chunk["type"] == "image":
                # Convert image to text description, then embed
                print(f"      Describing image on {chunk['source']} …")
                try:
                    description = self._client.describe_image(chunk["content"])
                except Exception as exc:
                    print(f"      [warning] Image description failed: {exc}", file=sys.stderr)
                    description = "(image — description unavailable)"
                text_content = IMAGE_DESCRIPTION_PREFIX + description
                meta = {
                    "type": "image_description",
                    "content": text_content,
                    "page": chunk["page"],
                    "source": chunk["source"],
                }
            else:
                text_content = chunk["content"]
                meta = {k: v for k, v in chunk.items()}

            try:
                embedding = self._client.embed_text(text_content)
                self._store.add(embedding, meta)
                embedded += 1
            except Exception as exc:
                print(
                    f"      [warning] Embedding failed for {chunk['source']}: {exc}",
                    file=sys.stderr,
                )

        print(f"      Embedded {embedded} chunks.")
        print("[3/3] Ingestion complete.")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = TOP_K) -> str:
        """
        Answer *question* using content retrieved from the indexed PDF.
        """
        if len(self._store) == 0:
            raise RuntimeError("Index is empty — ingest a PDF first.")

        # Embed the question
        q_embedding = self._client.embed_text(question, task_type="RETRIEVAL_QUERY")

        # Retrieve top-k relevant chunks
        results = self._store.search(q_embedding, top_k=top_k)

        # Build context string
        context_parts: List[str] = []
        for score, meta in results:
            source = meta.get("source", "unknown")
            content = meta.get("content", "")
            context_parts.append(f"[Source: {source} | score: {score:.3f}]\n{content}")
        context = "\n\n---\n\n".join(context_parts)

        # Generate answer
        return self._client.generate_answer(context, question)

    # ------------------------------------------------------------------
    # Persistence helpers (delegate to VectorStore)
    # ------------------------------------------------------------------

    def save_index(self, path: str) -> None:
        self._store.save(path)

    def load_index(self, path: str) -> None:
        self._store = VectorStore.load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF RAG pipeline powered by Gemini Embedding (text-embedding-004).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Ingest a PDF and save the index
              python pdf_rag.py ingest report.pdf --index report.json

              # Query a saved index
              python pdf_rag.py query "Summarise the key findings" --index report.json

              # Ingest and query in one shot (index is kept in memory)
              python pdf_rag.py ask report.pdf "What tables are present?"
            """
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Ingest a PDF and save the index.")
    p_ingest.add_argument("pdf", help="Path to the PDF file.")
    p_ingest.add_argument(
        "--index", default="rag_index.json", help="Output index file (default: rag_index.json)."
    )

    # --- query ---
    p_query = sub.add_parser("query", help="Query a saved index.")
    p_query.add_argument("question", help="Question to ask.")
    p_query.add_argument(
        "--index", default="rag_index.json", help="Index file to load (default: rag_index.json)."
    )
    p_query.add_argument(
        "--top-k", type=int, default=TOP_K, help=f"Number of chunks to retrieve (default: {TOP_K})."
    )

    # --- ask (one-shot) ---
    p_ask = sub.add_parser("ask", help="Ingest a PDF and immediately answer a question.")
    p_ask.add_argument("pdf", help="Path to the PDF file.")
    p_ask.add_argument("question", help="Question to ask.")
    p_ask.add_argument(
        "--index",
        default=None,
        help="Optionally save the index to this file after ingestion.",
    )
    p_ask.add_argument(
        "--top-k", type=int, default=TOP_K, help=f"Number of chunks to retrieve (default: {TOP_K})."
    )

    return parser


def _get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "Error: GEMINI_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  export GEMINI_API_KEY='your-api-key-here'"
        )
    return api_key


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    api_key = _get_api_key()
    rag = RAGPipeline(api_key)

    if args.command == "ingest":
        rag.ingest(args.pdf)
        rag.save_index(args.index)

    elif args.command == "query":
        rag.load_index(args.index)
        answer = rag.query(args.question, top_k=args.top_k)
        print("\n=== Answer ===")
        print(answer)

    elif args.command == "ask":
        rag.ingest(args.pdf)
        if args.index:
            rag.save_index(args.index)
        answer = rag.query(args.question, top_k=args.top_k)
        print("\n=== Answer ===")
        print(answer)


if __name__ == "__main__":
    main()
