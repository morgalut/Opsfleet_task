# src/helper_agent/prepare_docs.py
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from .config import load_config
from .llm_client import LLMClient


# Utility – stable hash for deduplication
def fast_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


# Load documentation files
def load_raw_files(raw_docs_dir: str) -> List[Path]:
    raw_dir = Path(raw_docs_dir)

    if not raw_dir.exists():
        raise RuntimeError(f"Folder {raw_docs_dir} does not exist!")

    files = list(raw_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(
            f"No .txt files found in {raw_docs_dir}. "
            "Download llms.txt / llms-full.txt first."
        )

    return files


# Gemini-Safe Chunker (never exceeds 30k bytes)
def chunk_text(text: str, max_bytes: int = 28000, split_bytes: int = 24000) -> List[str]:
    chunks: List[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    def size(s: str) -> int:
        return len(s.encode("utf-8"))

    for para in paragraphs:

        # Paragraph too large? Hard-split
        if size(para) > max_bytes:
            raw = para.encode("utf-8")
            for i in range(0, len(raw), split_bytes):
                piece = raw[i : i + split_bytes].decode("utf-8", errors="ignore").strip()
                if piece:
                    chunks.append(piece)
            continue

        # Try merging with previous chunk
        if chunks:
            candidate = chunks[-1] + "\n\n" + para
            if size(candidate) <= max_bytes:
                chunks[-1] = candidate
                continue

        # Otherwise start a new chunk
        chunks.append(para)

    return chunks


# Main index builder — saves ONLY embeddings.npy + docs.jsonl
def build_index(config):

    os.makedirs(config.index_dir, exist_ok=True)

    emb_path = Path(config.index_dir) / "embeddings.npy"
    docs_path = Path(config.index_dir) / "docs.jsonl"

    # Skip if index already exists
    if emb_path.exists() and docs_path.exists():
        print("✔ Index already exists. Skipping rebuild.")
        return

    print("Initializing Gemini…")
    llm = LLMClient(config)

    files = load_raw_files(config.raw_docs_dir)

    print(f"\nProcessing {len(files)} documentation files…\n")

    EMB_DIM = 768
    embeddings_list = []       # final matrix to save as .npy
    docs_out = docs_path.open("w", encoding="utf-8")

    seen_hashes = set()
    total_raw = 0
    total_unique = 0

    for f in files:
        print(f"[{f.name}]")

        text = f.read_text(encoding="utf-8")

        # Chunk
        chunks = chunk_text(text)
        total_raw += len(chunks)
        print(f"  → {len(chunks)} chunks before dedup")

        unique_chunks = []
        for c in chunks:
            if len(c) < 150:
                continue
            h = fast_hash(c)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_chunks.append(c)

        print(f"  → {len(unique_chunks)} unique chunks kept")
        total_unique += len(unique_chunks)

        # Embed each chunk
        for chunk in tqdm(unique_chunks, desc=f"Embedding {f.name}"):

            vec = llm.embed([chunk])[0]  # 1 embedding
            embeddings_list.append(vec)

            docs_out.write(
                json.dumps(
                    {"content": chunk, "metadata": {"file": f.name}},
                    ensure_ascii=False
                ) + "\n"
            )

    docs_out.close()

    # Convert list → numpy array (clean .npy)
    print("\nSaving embeddings.npy…")
    embedding_matrix = np.array(embeddings_list, dtype="float32")
    np.save(emb_path, embedding_matrix)

    print("\n✔ Indexing complete!")
    print(f"Raw chunks:        {total_raw}")
    print(f"Unique kept:       {total_unique}")
    print(f"Embeddings stored: {embedding_matrix.shape[0]}")
    print(f"Embeddings file:   {emb_path}")
    print(f"Docs JSONL:        {docs_path}")


def main():
    config = load_config(cli_mode="offline")
    build_index(config)


if __name__ == "__main__":
    main()
