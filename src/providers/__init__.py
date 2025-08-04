"""
LLM Providers package.

This package contains implementations of various LLM providers that can be used
with the test generation system.
"""

import importlib
from typing import List, Type, Any, Optional

from .default_provider import DefaultLLMProvider

# Try to import LangChain provider, but make it optional
try:
    from .langchain_provider import LangChainOllamaProvider
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create a dummy class for type hints when langchain is not available
    class LangChainOllamaProvider:  # type: ignore
        pass

# Only include LangChain provider in __all__ if it's available
__all__: List[str] = ['DefaultLLMProvider']

if LANGCHAIN_AVAILABLE:
    __all__.append('LangChainOllamaProvider')

def get_available_providers() -> List[str]:
    """Get a list of available provider names."""
    providers = ['default']
    if LANGCHAIN_AVAILABLE:
        providers.append('langchain_ollama')
    return providers
