"""
Configuration for LangChain integration.

This module provides configuration classes for LangChain and Ollama integration
while maintaining compatibility with the existing configuration system.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LangChainOllamaConfig:
    """Configuration for LangChain Ollama provider."""
    
    model_name: str ="qwen2.5-coder:7b"
    #model_name: str ="llama3.1:8b-instruct"
    # Model configuration
    #model_name: str = "codellama:instruct"
    temperature: float = 0.2
    max_tokens: int = 1000
    top_p: float = 0.9
    num_ctx: int = 4096
    
    # Server configuration
    base_url: str = "http://localhost:11434"
    
    # Advanced settings
    num_gpu: int = 1  # Set to 0 for CPU-only
    num_thread: int = 8
    timeout: int = 120  # seconds
    keep_alive: str = "10m"  # Keep the model loaded in Ollama
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
            "base_url": self.base_url,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "timeout": self.timeout,
            "keep_alive": self.keep_alive,
        }

@dataclass
class LangChainConfig:
    """Top-level LangChain configuration."""
    
    # Provider configuration
    provider: str = "langchain_ollama"  # Could support other providers in the future
    
    # Ollama-specific configuration
    ollama: LangChainOllamaConfig = field(default_factory=LangChainOllamaConfig)
    
    # Whether to enable LangChain integration
    enabled: bool = True
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "provider": self.provider,
            "enabled": self.enabled,
            "ollama": self.ollama.to_dict()
        }

# Default configuration
default_config = LangChainConfig()
