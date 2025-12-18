from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import AppConfig
from src.llm.client import LLMClient, OpenAICompatibleChatClient


@dataclass(frozen=True, slots=True)
class LLMFactory:
    """
    Provider-based LLM client factory.

    Rationale:
    - Selection is driven by config (`LLM_PROVIDER`)
    - Provider-specific config validation happens at config load time (deploy-time),
      so this factory can assume required values exist when provider is enabled.
    """

    cfg: AppConfig

    def create(self) -> Optional[LLMClient]:
        provider = (self.cfg.llm_provider or "none").strip().lower()
        if provider == "none":
            return None
        if provider == "openai_compatible":
            return OpenAICompatibleChatClient(self.cfg)
        raise RuntimeError(f"Unsupported LLM provider: {provider}")


def create_llm_client(cfg: AppConfig) -> Optional[LLMClient]:
    return LLMFactory(cfg).create()


