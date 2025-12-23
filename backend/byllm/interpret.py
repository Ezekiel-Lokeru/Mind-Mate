"""byLLM interpreter using pluggable adapter."""
from typing import Dict, Any

from .adapter import LLMAdapter

_adapter = LLMAdapter()


def interpret_input(text: str) -> Dict[str, Any]:
    """Interpret free-text mood input and extract emotions, triggers, and intensity."""
    return _adapter.interpret(text)