# src/helper_agent/llm_client.py
from __future__ import annotations

import os
import time
import random
from typing import List, Optional, Tuple

import google.generativeai as genai

from .config import AppConfig


class LLMClient:
    """
    Improved Gemini LLM client with:
      - automatic model selection
      - retry on errors
      - retry on truncation
      - reasoning-mode control via AGENT_REASONING
      - safer embedding fallback
      - max token auto-expansion
    """

    # ============================================================
    # INIT
    # ============================================================
    def __init__(self, config: AppConfig) -> None:
        self.config = config

        # Set reasoning mode: full | lite | off
        self.reasoning_mode = os.getenv("AGENT_REASONING", "full").lower()

        # Configure Gemini
        genai.configure(api_key=config.gemini_api_key)

        # List available models
        try:
            available_models = [m.name for m in genai.list_models()]
        except Exception:
            available_models = []
            print("[LLM] WARNING: Unable to list available Gemini models.")

        # Preferred model order
        preferred = [
            "models/gemini-2.0-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro",
        ]

        # Choose best available model
        chosen_text_model: Optional[str] = None
        for m in preferred:
            if m in available_models:
                chosen_text_model = m
                break

        # Fallback: pick *any* generative Gemini model
        if chosen_text_model is None:
            for m in available_models:
                # any gemini model that is not an embedding model
                if "gemini" in m.lower() and "embed" not in m.lower():
                    chosen_text_model = m
                    break

        config.gemini_model = chosen_text_model

        # Embedding model
        if "models/text-embedding-004" in available_models:
            config.embedding_model = "models/text-embedding-004"
        else:
            for m in available_models:
                if "embed" in m.lower():
                    config.embedding_model = m
                    break

        print("\n[LLM] Loaded models:")
        print(f"  - Text: {config.gemini_model}")
        print(f"  - Embedding: {config.embedding_model}\n")

        # Load generative model
        if config.gemini_model:
            self._model = genai.GenerativeModel(model_name=config.gemini_model)
        else:
            raise RuntimeError("No valid Gemini text model found!")

        # default max tokens (can be overridden by config)
        self.max_output_tokens: int = getattr(config, "max_output_tokens", 2048)

    # ============================================================
    # PUBLIC: GENERATE (with retry + truncation recovery)
    # ============================================================
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        High-level generation with:
          - up to 4 retries
          - exponential backoff on errors
          - automatic retry when output is truncated
        """
        max_tokens = max_output_tokens or self.max_output_tokens

        for attempt in range(1, 5):  # up to 4 attempts
            try:
                text, truncated = self._generate_once(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                if truncated:
                    # bump token limit and try again
                    print(
                        f"[LLM] Output truncated at {max_tokens} tokens "
                        f"→ retrying with {max_tokens * 2}"
                    )
                    max_tokens *= 2
                    continue

                return text

            except Exception as e:
                if attempt == 4:
                    print(f"[LLM] FATAL: final failure after 4 attempts → {e}")
                    return f"(LLM failure: {e})"

                delay = (2 ** (attempt - 1)) + random.uniform(0, 0.4)
                print(f"[LLM] Error: {e} → retrying in {delay:.2f}s")
                time.sleep(delay)

        return "(no output)"

    # ============================================================
    # INTERNAL: single Gemini call
    # ============================================================
    def _generate_once(
        self,
        prompt: str,
        system_instruction: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, bool]:
        """
        Low-level single Gemini call. Returns (text, truncated_flag).
        """
        # Reasoning mode injection
        if self.reasoning_mode == "lite":
            prompt += "\n\n(Keep chain-of-thought extremely brief.)"
        elif self.reasoning_mode == "off":
            prompt += "\n\n(Do NOT output chain-of-thought. Provide only the final answer.)"

        parts: List[str] = []
        if system_instruction:
            parts.append(system_instruction)
        parts.append(prompt)

        response = self._model.generate_content(
            parts,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        text = ""
        if response and getattr(response, "candidates", None):
            first = response.candidates[0]
            for p in first.content.parts:
                if hasattr(p, "text") and p.text:
                    text += p.text

            finish_reason = getattr(first, "finish_reason", None)
            truncated = finish_reason == "MAX_TOKENS"
        else:
            truncated = False

        return text.strip(), truncated

    # ============================================================
    # EMBEDDINGS
    # ============================================================
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Google Gemini does NOT support true batch embedding.
        So we loop manually but keep the same interface.
        """
        vectors: List[List[float]] = []

        for text in texts:
            try:
                r = genai.embed_content(
                    model=self.config.embedding_model,
                    content=text,
                )
                vectors.append(r["embedding"])
            except Exception as e:
                print(f"[embed error] Skipping chunk ({len(text)} chars): {e}")
                # append zero vector to keep index aligned
                vectors.append([0.0] * 768)

        return vectors
