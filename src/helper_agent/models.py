# src/helper_agent/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Literal, TypedDict


@dataclass
class Document:
    """
    A simple representation of a retrieved document chunk.
    """
    content: str
    source: str
    metadata: dict[str, Any]


class AgentState(TypedDict, total=False):
    # Core
    question: str
    mode: Literal["offline", "online"]

    # Retrieval / answer
    retrieved_docs: List[Document]
    draft_answer: str
    final_answer: str
    error: Optional[str]

    # Reasoning style selector: 'basic' | 'orc' | 'react' | 'hybrid'
    reasoning_style: Literal["basic", "orc", "react", "hybrid"]

    # --- ORC-specific fields ---
    orc_plan: str
    orc_subquestions: List[str]

    # --- ReAct-specific fields ---
    scratchpad: str          # Thought/Action/Observation log
    react_mode: Literal["act", "finish"]
    react_query: str         # current action query for search
    iterations: int          # loop counter to avoid infinite loops
