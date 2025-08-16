"""
LLM service implementation for AI-powered code generation.

This module provides a production-ready LLM service that can interface with
multiple LLM backends, including local models via LangChain/Ollama.
"""

import time
import requests
from typing import Dict, Any, Optional, Union, Type, TypeVar
from dataclasses import dataclass
import importlib

from src.interfaces.base_interfaces import LLMProvider
from src.models.data_models import ModelMetrics
from src.config.settings import LLMConfig
from src.config.langchain_config import LangChainConfig, default_config as default_langchain_config
from src.providers.default_provider import DefaultLLMProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for provider classes
T = TypeVar('T', bound=LLMProvider)


@dataclass
class LLMResponse:
    """Response from LLM service."""
    success: bool
    text: str
    response_time: float
    error_message: Optional[str] = None
    token_count: Optional[int] = None
    model_used: Optional[str] = None
    provider: Optional[str] = None  # Track which provider was used


class LLMService(LLMProvider):
    """
    Production-ready LLM service with comprehensive error handling.
    
    Features:
    - Multiple provider support (OpenAI, LangChain/Ollama, etc.)
    - Robust error handling and retries
    - Response time monitoring
    - Token usage tracking
    - Configurable generation parameters
    - Health check capabilities
    - Detailed logging and metrics
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, langchain_config: Optional[LangChainConfig] = None):
        """
        Initialize the LLM service.
        
        Args:
            config: LLM configuration
            langchain_config: LangChain-specific configuration
        """
        from src.config.settings import config as app_config
        
        self.config = config or LLMConfig()
        self.langchain_config = langchain_config or getattr(app_config, 'langchain', default_langchain_config)
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize the appropriate provider
        self.provider = self._initialize_provider()
        
        # Metrics tracking
        self.metrics = ModelMetrics(model_name=self.config.model_name)
        
        self.logger.info(f"Initialized LLMService with provider: {type(self.provider).__name__}")
    
    def _initialize_provider(self) -> LLMProvider:
        """
        Initialize the appropriate LLM provider based on configuration.
        
        Returns:
            An instance of a class implementing the LLMProvider interface
        """
        from src.config.settings import config as app_config
        from src.providers import LANGCHAIN_AVAILABLE, get_available_providers
        
        # Check if we should use LangChain provider
        use_langchain = getattr(self.config, 'use_langchain', False) or getattr(app_config.model, 'use_langchain', False)
        langchain_provider = getattr(self.config, 'langchain_provider', None) or getattr(app_config.model, 'langchain_provider', 'langchain_ollama')
        
        self.logger.debug(f"Provider initialization - use_langchain: {use_langchain}, langchain_provider: {langchain_provider}")
        self.logger.debug(f"LANGCHAIN_AVAILABLE: {LANGCHAIN_AVAILABLE}")
        self.logger.debug(f"Available providers: {get_available_providers()}")
        self.logger.debug(f"Config: {self.config}")
        self.logger.debug(f"App config model: {app_config.model}")
        
        # If LangChain is requested but not available, log a warning
        if use_langchain and not LANGCHAIN_AVAILABLE:
            self.logger.warning(
                "LangChain provider was requested but is not available. "
                "Falling back to default provider. Available providers: "
                f"{', '.join(get_available_providers())}"
            )
            use_langchain = False
        
        # If LangChain is requested and available, try to initialize it
        if use_langchain and LANGCHAIN_AVAILABLE:
            self.logger.info("Attempting to initialize LangChain provider")
            self.logger.debug(f"Available providers: {get_available_providers()}")
        
        try:
            if use_langchain and langchain_provider == 'langchain_ollama':
                self.logger.info("Initializing LangChain Ollama provider")
                from src.providers import LangChainOllamaProvider
                
                # Get Ollama configuration
                ollama_config = self.langchain_config.ollama
                
                # Create provider with configuration from settings
                return LangChainOllamaProvider(
                    model_name=ollama_config.model_name,
                    temperature=ollama_config.temperature,
                    max_tokens=ollama_config.max_tokens,
                    top_p=ollama_config.top_p,
                    num_ctx=ollama_config.num_ctx,
                    base_url=ollama_config.base_url,
                    num_gpu=ollama_config.num_gpu,
                    num_thread=ollama_config.num_thread,
                    timeout=ollama_config.timeout,
                    keep_alive=getattr(ollama_config, 'keep_alive', '10m')
                )
            
            # Fall back to default provider
            self.logger.info("Initializing default LLM provider")
            return DefaultLLMProvider(self.config)
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize LLM provider: {e}", 
                exc_info=True
            )
            self.logger.info("Falling back to default LLM provider")
            return DefaultLLMProvider(self.config)
        
        self.logger.info(f"Initialized LLMService with config: {self.config}")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text using the configured LLM provider.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with the generated text and metadata
        """
        start_time = time.time()
        response = LLMResponse(
            success=False,
            text="",
            response_time=0.0,
            model_used=self.config.model_name,
            provider=type(self.provider).__name__
        )
        
        try:
            # Delegate to the provider
            generated_text = self.provider.generate(prompt, **kwargs)
            
            # Update response
            response.text = generated_text
            response.success = True
            
        except Exception as e:
            response.error_message = f"Generation failed: {str(e)}"
            self.logger.error(response.error_message)
        
        # Update metrics
        response.response_time = time.time() - start_time
        self._update_metrics(response.response_time, response.success)
        
        return response
    
    def is_available(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            # Simple health check with minimal prompt
            health_check_prompt = "Hello"
            response = self.generate(health_check_prompt)
            return bool(response.text)
            
        except Exception as e:
            self.logger.warning(f"LLM service health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'api_url': self.config.api_url,
            'model_name': self.config.model_name,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'is_available': self.is_available()
        }
    
    def _prepare_request_params(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters for API call."""
        params = {
            'model': self.config.model_name,
            'prompt': prompt,
            'stream': False,
            'temperature': kwargs.get('temperature', self.config.temperature),
            'top_p': kwargs.get('top_p', self.config.top_p),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        return params
    
    def _make_api_request(self, params: Dict[str, Any]) -> LLMResponse:
        """Make the actual API request to Ollama."""
        start_time = time.time()
        
        try:
            response = requests.post(
                self.config.api_url,
                json=params,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data.get('response', '').strip()
                
                return LLMResponse(
                    success=True,
                    text=generated_text,
                    response_time=response_time,
                    model_used=params.get('model')
                )
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                return LLMResponse(
                    success=False,
                    text="",
                    response_time=response_time,
                    error_message=error_msg
                )
                
        except requests.exceptions.Timeout:
            return LLMResponse(
                success=False,
                text="",
                response_time=self.config.timeout,
                error_message="Request timeout"
            )
        except requests.exceptions.ConnectionError:
            return LLMResponse(
                success=False,
                text="",
                response_time=time.time() - start_time,
                error_message="Connection error - is Ollama server running?"
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                text="",
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _update_metrics(self, response_time: float, success: bool):
        """Update service metrics using the ModelMetrics class.
        
        Args:
            response_time: Time taken for the request in seconds
            success: Whether the request was successful
        """
        try:
            # Get token count if available from the provider
            token_count = None
            if hasattr(self.provider, 'last_token_count'):
                token_count = self.provider.last_token_count
                
            # Update metrics using the ModelMetrics class
            self.metrics.update_metrics(
                response_time=response_time,
                success=success,
                tokens_used=token_count
            )
            
            # Log the metrics update
            self.logger.debug(
                f"Updated metrics - Success: {success}, "
                f"Response Time: {response_time:.2f}s, "
                f"Token Count: {token_count or 'N/A'}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}", exc_info=True)


# Legacy compatibility class for backward compatibility
class LLMClient:
    """
    Legacy wrapper for backward compatibility with existing code.
    
    Note: This class is deprecated. Use LLMService instead.
    """
    
    def __init__(self, api_url: str, model_name: str):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.warning("Using deprecated LLMClient. Consider updating to LLMService.")
        
        # Create config from legacy parameters
        from src.config.settings import LLMConfig
        config = LLMConfig(
            api_url=api_url,
            model_name=model_name
        )
        
        # Initialize new service
        self.service = LLMService(config)
    
    def generate(self, prompt: str) -> str:
        """Legacy method for generating text."""
        return self.service.generate(prompt)
