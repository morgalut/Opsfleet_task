# src/helper_agent/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------
# Types
# -----------------------------------------------------
Mode = Literal["offline", "online"]
Reasoning = Literal["basic", "orc", "react", "hybrid"]
SearchProvider = Literal["duckduckgo", "none"]


# -----------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------
@dataclass
class AppConfig:
    # Core mode
    mode: Mode
    gemini_api_key: str

    # Reasoning strategy
    reasoning_style: Reasoning = "basic"

    # Search provider
    search_provider: SearchProvider = "duckduckgo"
    search_api_key: Optional[str] = None  # no API key needed for DDG

    # Index + data paths
    data_dir: str = "data"
    raw_docs_dir: str = "data/raw"
    index_dir: str = "data/index"

    # Model names (resolved dynamically)
    gemini_model: Optional[str] = None
    embedding_model: Optional[str] = None

    # Output max tokens (needed by LLMClient)
    max_output_tokens: int = 2048


# -----------------------------------------------------
# Config Loader
# -----------------------------------------------------
def load_config(
    cli_mode: Optional[str] = None,
    cli_reasoning: Optional[str] = None,
) -> AppConfig:

    # --- MODE -------------------------------------------------
    mode_str = cli_mode or os.getenv("AGENT_MODE", "offline").lower()
    if mode_str not in ("offline", "online"):
        raise ValueError("AGENT_MODE must be 'offline' or 'online'.")
    mode: Mode = mode_str  # type: ignore

    # --- GEMINI KEY -------------------------------------------
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError(
            "❌ Missing GEMINI_API_KEY.\n"
            "Create one at https://aistudio.google.com\n"
            "Add to .env:\n\n"
            "  GEMINI_API_KEY=your_key_here\n"
        )

    # --- SEARCH PROVIDER --------------------------------------
    sp = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()
    if sp not in ("duckduckgo", "none"):
        raise ValueError("SEARCH_PROVIDER must be 'duckduckgo' or 'none'.")
    search_provider: SearchProvider = sp  # type: ignore

    # --- REASONING STYLE --------------------------------------
    rs = cli_reasoning or os.getenv("AGENT_REASONING", "basic").lower()
    if rs not in ("basic", "orc", "react", "hybrid"):
        raise ValueError("Reasoning must be: basic, orc, react, hybrid.")
    reasoning_style: Reasoning = rs  # type: ignore

    # --- CREATE CONFIG OBJECT ---------------------------------
    cfg = AppConfig(
        mode=mode,
        gemini_api_key=gemini_api_key,
        search_provider=search_provider,
        search_api_key=None,
        reasoning_style=reasoning_style,
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "2048")),
    )

    # --- DIAGNOSTICS ------------------------------------------
    print("\n[CONFIG]")
    print(f"  Mode:            {cfg.mode}")
    print(f"  Reasoning:       {cfg.reasoning_style}")
    print(f"  Search Provider: {cfg.search_provider}")
    print(f"  Max Output:      {cfg.max_output_tokens}")
    print("  No search API keys required.\n")

    return cfg
