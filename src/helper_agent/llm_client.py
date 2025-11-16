# src/helper_agent/llm_client.py
from __future__ import annotations
from typing import List, Optional
import google.generativeai as genai

from .config import AppConfig


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

        genai.configure(api_key=config.gemini_api_key)

        # ----------------------------------------------------------
        # AUTO-DETECT AVAILABLE MODELS (this prevents 404 errors)
        # ----------------------------------------------------------
        available = [m.name for m in genai.list_models()]

        # preferred models in order
        preferred = [
            "models/gemini-2.0-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro",
        ]

        for m in preferred:
            if m in available:
                config.gemini_model = m
                break

        # fallback: first generative text model
        if config.gemini_model is None:
            for m in available:
                if "gemini" in m and "embed" not in m:
                    config.gemini_model = m
                    break

        # --- embedding model ---
        if "models/text-embedding-004" in available:
            config.embedding_model = "models/text-embedding-004"
        else:
            # fallback: first embedding model
            for m in available:
                if "embed" in m.lower():
                    config.embedding_model = m
                    break

        print("\n[LLM] Loaded models:")
        print("  - Text:", config.gemini_model)
        print("  - Embedding:", config.embedding_model, "\n")

        # now load generative model
        self._model = genai.GenerativeModel(model_name=config.gemini_model)

    # --------------------------------------------------------------
    # TEXT GENERATION
    # --------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:

        parts = []
        if system_instruction:
            parts.append(system_instruction)
        parts.append(prompt)

        response = self._model.generate_content(
            parts,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            },
        )

        text = ""
        for p in response.candidates[0].content.parts:
            if hasattr(p, "text") and p.text:
                text += p.text

        return text.strip()

    # --------------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------------
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Google Gemini does NOT support true batch embedding.
        So we loop manually but keep the same interface.
        """
        vectors = []

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
