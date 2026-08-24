"""Provider-agnostic LLM access."""

from .router import Tier, call_llm, embed, get_client

__all__ = ["Tier", "call_llm", "embed", "get_client"]
