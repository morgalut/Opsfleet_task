# src/helper_agent/retrievers.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import AppConfig
from .llm_client import LLMClient
from .models import Document


# ================================================================
# Base Interface
# ================================================================
class DocRetriever:
    """Abstract base class."""
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        raise NotImplementedError


# ================================================================
# OFFLINE RETRIEVER (VECTOR RAG)
# ================================================================
class OfflineDocRetriever(DocRetriever):
    """
    Retrieves documentation chunks from:
      - embeddings.npy
      - docs.jsonl
    Offline mode only.
    """

    def __init__(self, config: AppConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

        emb_path = Path(config.index_dir) / "embeddings.npy"
        docs_path = Path(config.index_dir) / "docs.jsonl"

        if not emb_path.exists() or not docs_path.exists():
            raise RuntimeError(
                f"Offline index missing. Run:\n"
                f"python -m src.helper_agent.prepare_docs"
            )

        # Load + normalize embeddings
        self.embeddings = np.load(emb_path)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        self.embeddings = self.embeddings / norms

        # Load docs
        docs = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    docs.append(json.loads(line))
                except:
                    continue
        self.docs = docs

        if len(self.docs) != len(self.embeddings):
            raise RuntimeError("Index corrupted: embeddings/docs size mismatch.")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        q_vec = np.array(self.llm.embed([query])[0], dtype="float32")
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

        scores = self.embeddings @ q_vec
        top_idx = np.argsort(scores)[::-1][:k]

        out = []
        for idx in top_idx:
            row = self.docs[idx]
            out.append(
                Document(
                    content=row["content"],
                    source=row["metadata"].get("file", "local"),
                    metadata={"score": float(scores[idx])},
                )
            )
        return out


# ================================================================
# ONLINE RETRIEVER (TAVILY + fallback + scoring)
# ================================================================
class OnlineDocRetriever(DocRetriever):
    """
    Unified online retriever with:
      - Tavily (if available)
      - Clean DDG fallback
      - Domain + keyword scoring
      - Offline RAG fallback
      - ZERO repeated warnings per run
    """

    GOOD_DOMAINS = [
        "langchain-ai.github.io",
        "python.langchain.com",
        "api.python.langchain.com",
        "langgraph.readthedocs.io",
        "github.com/langchain-ai",
    ]

    KEYWORDS = [
        "langgraph", "langchain", "node", "state",
        "retry", "workflow", "persistence", "branching", "graph"
    ]

    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self.provider = config.search_provider

        # --- NEW flags: show warnings only once per run ---
        self.warned_ddg = False
        self.warned_quality = False

        # Try Tavily
        self.use_tavily = False
        if self.provider == "tavily" and config.search_api_key:
            try:
                from tavily import TavilyClient
                self.tavily = TavilyClient(api_key=config.search_api_key)
                self.use_tavily = True
                print("[OnlineDocRetriever] Using Tavily Search API.")
            except Exception as e:
                print(f"[WARN] Tavily import failed ({e}) → using DDG fallback.")

        # Try DDG
        self.ddg = None
        if not self.use_tavily:
            try:
                from ddgs import DDGS
                self.ddg = DDGS()
                print("[OnlineDocRetriever] Using ddgs search backend.")
            except Exception:
                try:
                    from duckduckgo_search import DDGS
                    self.ddg = DDGS()
                    print("[OnlineDocRetriever] Using duckduckgo_search backend.")
                except Exception:
                    print("[WARN] DDG unavailable → using offline fallback.")

        # Offline RAG fallback
        try:
            self.offline = OfflineDocRetriever(config, llm)
        except Exception:
            self.offline = None
            print("[WARN] Offline index unavailable.")

    # --------------------------
    # Score result
    # --------------------------
    def _score_result(self, text: str, url: str) -> float:
        score = 0

        for dom in self.GOOD_DOMAINS:
            if dom in url:
                score += 4

        text_lower = text.lower()
        for kw in self.KEYWORDS:
            if kw in text_lower:
                score += 1.5

        if len(text) > 200:
            score += 1

        return score

    # --------------------------
    # Main search
    # --------------------------
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        boosted_query = (
            f"{query} site:langchain-ai.github.io "
            "OR site:python.langchain.com "
            "OR site:api.python.langchain.com "
            "OR site:langgraph.readthedocs.io"
        )

        docs = []

        # 1. Tavily (preferred)
        if self.use_tavily:
            try:
                results = self.tavily.search(boosted_query, max_results=8)
                for r in results.get("results", []):
                    content = (r.get("content") or "").strip()
                    url = r.get("url", "")
                    score = self._score_result(content, url)
                    docs.append(Document(
                        content=content,
                        source=url,
                        metadata={"title": r.get("title", ""), "score": score},
                    ))
            except Exception:
                if not self.warned_ddg:
                    print("[WARN] Tavily failed → switching to DDG.")
                    self.warned_ddg = True

        # 2. DuckDuckGo fallback
        if not docs and self.ddg:
            try:
                results = list(self.ddg.text(boosted_query, max_results=8) or [])
                for r in results:
                    text = r.get("body") or r.get("snippet") or ""
                    url = r.get("href", "")
                    score = self._score_result(text, url)
                    docs.append(Document(
                        content=text,
                        source=url,
                        metadata={"title": r.get("title", ""), "score": score},
                    ))
            except Exception:
                if not self.warned_ddg:
                    print("[WARN] DDG search failed → using offline fallback.")
                    self.warned_ddg = True

        # 3. Sort by quality
        docs = sorted(docs, key=lambda d: d.metadata.get("score", 0), reverse=True)

        # 4. Low-quality → offline fallback
        if not docs or docs[0].metadata.get("score", 0) < 2.0:
            if not self.warned_quality:
                print("[INFO] Online results low quality → using offline RAG fallback.")
                self.warned_quality = True
            return self._offline_fallback(query, k)

        return docs[:k]

    # --------------------------
    # Offline fallback
    # --------------------------
    def _offline_fallback(self, query: str, k: int) -> List[Document]:
        if not self.offline:
            return [Document("(no search available)", "none", {})]

        docs = self.offline.retrieve(query, k)
        for d in docs:
            d.metadata["fallback"] = True
            d.metadata["confidence"] = 0.9
        return docs
