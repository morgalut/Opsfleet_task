# src/helper_agent/retrievers.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .config import AppConfig
from .llm_client import LLMClient
from .models import Document


# ================================================================
# Base Interface
# ================================================================
class DocRetriever:
    """
    Abstract base class for offline/online retrievers.
    """
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        raise NotImplementedError


# ================================================================
# OFFLINE RETRIEVER (REAL RAG)
# ================================================================
class OfflineDocRetriever(DocRetriever):
    """
    Retrieves documentation chunks from the local vector index using:
      - embeddings.npy  (float32 matrix)
      - docs.jsonl      (one JSON per line)

    This is used in offline mode: local docs only, no internet.
    """

    def __init__(self, config: AppConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

        emb_path = Path(config.index_dir) / "embeddings.npy"
        docs_path = Path(config.index_dir) / "docs.jsonl"

        # ----------------------------
        # Validation
        # ----------------------------
        if not emb_path.exists():
            raise RuntimeError(
                f"Offline index missing embeddings file:\n{emb_path}\n"
                "Run: python -m src.helper_agent.prepare_docs"
            )

        if not docs_path.exists():
            raise RuntimeError(
                f"Offline index missing docs file:\n{docs_path}\n"
                "Run: python -m src.helper_agent.prepare_docs"
            )

        # ----------------------------
        # Load embeddings
        # ----------------------------
        self.embeddings = np.load(emb_path)  # shape (N, 768)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        self.embeddings = self.embeddings / norms

        # ----------------------------
        # Load docs from JSONL
        # ----------------------------
        docs = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[WARN] bad JSONL row skipped: {e}")
                        continue

        self.docs = docs

        # Defensive check
        if len(self.docs) != len(self.embeddings):
            raise RuntimeError(
                "Embedding matrix and docs.jsonl size mismatch.\n"
                f"Embeddings: {len(self.embeddings)} rows\n"
                f"Docs:       {len(self.docs)} rows\n"
                "Your index build did not complete cleanly."
            )

        print(
            f"[OfflineDocRetriever] Loaded {len(self.docs)} chunks "
            f"from {docs_path}"
        )

    # ------------------------------------------------------------
    # Query → embed → cosine similarity → top-k documents
    # ------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        # Embed query
        q_vec = np.array(self.llm.embed([query])[0], dtype="float32")
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

        # Cosine similarity
        scores = self.embeddings @ q_vec

        # Best k
        top_idx = np.argsort(scores)[::-1][:k]

        # Convert to Document objects
        out: List[Document] = []

        for idx in top_idx:
            row = self.docs[idx]
            out.append(
                Document(
                    content=row["content"],
                    source=row["metadata"].get("file", "offline_docs"),
                    metadata={"score": float(scores[idx])},
                )
            )

        return out


# ================================================================
# ONLINE RETRIEVER (DUCKDUCKGO)
# ================================================================
class OnlineDocRetriever(DocRetriever):
    """
    Retrieves documentation via online search (DuckDuckGo or Tavily).
    This is used in online mode (internet allowed).
    """

    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self.provider = config.search_provider

        if self.provider == "duckduckgo":
            from duckduckgo_search import DDGS  # type: ignore
            self.ddg = DDGS()
        else:
            # Tavily, SERP, EXA, etc. can be added here.
            self.ddg = None

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        docs: List[Document] = []

        # DuckDuckGo search
        if self.provider == "duckduckgo" and self.ddg is not None:
            results = self.ddg.text(query, max_results=k)
            for r in results:
                docs.append(
                    Document(
                        content=r.get("body") or r.get("snippet") or "",
                        source=r.get("href", ""),
                        metadata={"title": r.get("title", "")},
                    )
                )

        # If no results or provider disabled
        if not docs:
            docs.append(
                Document(
                    content="(online retriever not configured or returned no results)",
                    source="online_stub",
                    metadata={},
                )
            )

        return docs
