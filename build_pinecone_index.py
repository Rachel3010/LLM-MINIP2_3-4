"""
Part 3 setup: build Pinecone index from PDF (run once).
Uses the same index name and namespace as Part 3 Query_Agent, so you can run Part 3 without submitting Part 1.

Usage:
  1. Place machine-learning.pdf in the same directory as this script (or set PDF_PATH)
  2. Set env vars OPENAI_API_KEY, PINECONE_API_KEY
  3. Run: python build_pinecone_index.py
  4. Then run: streamlit run streamlit_app.py
"""
import os
import uuid

# PyMuPDF for reading PDF (pip install pymupdf)
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ---------- Config (must match Part 3) ----------
PDF_PATH = os.environ.get("PDF_PATH", "machine-learning.pdf")
INDEX_NAME = "machine-learning-textbook"
NAMESPACE = "ns2500"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 50
EMBED_MODEL = "text-embedding-3-small"
DIMENSION = 1536
BATCH_SIZE = 100


def load_pdf(path: str):
    """Load PDF text by page."""
    if fitz is None:
        raise ImportError("Install PyMuPDF: pip install pymupdf")
    doc = fitz.open(path)
    page_texts = []
    page_numbers = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        page_texts.append(page.get_text())
        page_numbers.append(i + 1)
    doc.close()
    return page_texts, page_numbers


def chunk_texts(page_texts, page_numbers, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split into chunks by character count with overlap (aligned with Part 1 logic)."""
    chunked_texts = []
    chunk_page_numbers = []
    previous_tail = ""
    for text, page_num in zip(page_texts, page_numbers):
        full = previous_tail + text
        start = 0
        while start < len(full):
            end = start + chunk_size
            chunk = full[start:end]
            chunked_texts.append(chunk)
            chunk_page_numbers.append(page_num)
            start = end - overlap
            if start >= len(full):
                break
        previous_tail = full[-overlap:] if len(full) > overlap else full
    return chunked_texts, chunk_page_numbers


def main():
    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        print("Set env vars OPENAI_API_KEY and PINECONE_API_KEY")
        return

    if not os.path.isfile(PDF_PATH):
        print(f"PDF not found: {PDF_PATH}")
        print("Place machine-learning.pdf in the current directory or set PDF_PATH")
        return

    print("1. Loading PDF...")
    page_texts, page_numbers = load_pdf(PDF_PATH)
    print(f"   Loaded {len(page_texts)} pages")

    print("2. Chunking...")
    chunked_texts, chunk_page_numbers = chunk_texts(page_texts, page_numbers)
    print(f"   {len(chunked_texts)} chunks")

    print("3. Generating embeddings...")
    client = OpenAI(api_key=openai_key)
    vectors_to_upsert = []
    for i, (text, page_num) in enumerate(zip(chunked_texts, chunk_page_numbers)):
        resp = client.embeddings.create(input=[text], model=EMBED_MODEL)
        vec = resp.data[0].embedding
        vectors_to_upsert.append({
            "id": str(uuid.uuid4()),
            "values": vec,
            "metadata": {"text": text, "page_number": page_num},
        })
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(chunked_texts)}")

    print("4. Connecting to Pinecone and creating index if needed...")
    pc = Pinecone(api_key=pinecone_key)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"   Created index: {INDEX_NAME}")
    else:
        print(f"   Index exists: {INDEX_NAME}")

    index = pc.Index(INDEX_NAME)
    print(f"5. Upserting to namespace={NAMESPACE}...")
    for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
        batch = vectors_to_upsert[i : i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=NAMESPACE)
        print(f"   Upserted {min(i + BATCH_SIZE, len(vectors_to_upsert))}/{len(vectors_to_upsert)}")
    print("Done. Run: streamlit run streamlit_app.py")
    return index


if __name__ == "__main__":
    main()
