# src/helper_agent/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Document:
    """
    A simple representation of a retrieved document chunk.
    """
    content: str
    source: str
    metadata: dict[str, Any]


# State for the LangGraph agent (we'll wire this in graph.py)
from typing import TypedDict, Literal


class AgentState(TypedDict, total=False):
    question: str
    mode: Literal["offline", "online"]
    retrieved_docs: List[Document]
    draft_answer: str
    final_answer: str
    error: Optional[str]
