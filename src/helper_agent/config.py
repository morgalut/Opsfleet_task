# src/helper_agent/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

# ✅ Load .env automatically
from dotenv import load_dotenv
load_dotenv()


Mode = Literal["offline", "online"]


@dataclass
class AppConfig:
    mode: Mode
    gemini_api_key: str

    # Search settings (online mode)
    search_provider: Literal["duckduckgo", "tavily", "none"] = "duckduckgo"
    search_api_key: Optional[str] = None

    # Offline paths
    data_dir: str = "data"
    raw_docs_dir: str = "data/raw"
    index_dir: str = "data/index"

    # Models – automatically selected in llm_client.py
    gemini_model: Optional[str] = None
    embedding_model: Optional[str] = None


def load_config(cli_mode: Optional[str] = None) -> AppConfig:
    """
    Load configuration and secrets from environment variables.
    """

    # --------------------------
    # 1. Load mode
    # --------------------------
    mode_str = cli_mode or os.environ.get("AGENT_MODE", "offline")
    if mode_str not in ("offline", "online"):
        raise ValueError("AGENT_MODE must be 'offline' or 'online'.")
    mode: Mode = mode_str  # type: ignore

    # --------------------------
    # 2. Load Gemini API key
    # --------------------------
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError(
            "❌ GEMINI_API_KEY is not set.\n"
            "   Create one at https://aistudio.google.com\n"
            "   And set it in your .env:\n\n"
            "   GEMINI_API_KEY=\"your-key-here\"\n"
        )

    # --------------------------
    # 3. Search provider
    # --------------------------
    search_provider = os.environ.get("SEARCH_PROVIDER", "duckduckgo")
    if search_provider not in ("duckduckgo", "tavily", "none"):
        raise ValueError("SEARCH_PROVIDER must be: duckduckgo, tavily, none.")

    search_api_key = os.environ.get("SEARCH_API_KEY")

    # --------------------------
    # 4. Build config object
    # --------------------------
    return AppConfig(
        mode=mode,
        gemini_api_key=gemini_api_key,
        search_provider=search_provider,  # type: ignore
        search_api_key=search_api_key,
    )
