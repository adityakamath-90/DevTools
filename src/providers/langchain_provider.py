"""
LangChain LLM Provider implementation for the test generation system.

This module provides a LangChain-based implementation of the LLMProvider interface,
allowing the use of local LLMs through Ollama while maintaining compatibility
with the existing codebase.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Type, TypeVar, Any

# Try to import LangChain dependencies, but make them optional
try:
    from langchain_community.llms import Ollama
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Create dummy classes for type hints when langchain is not available
    class Ollama:  # type: ignore
        pass
    
    class CallbackManager:  # type: ignore
        pass
    
    class StreamingStdOutCallbackHandler:  # type: ignore
        pass
    
    LANGCHAIN_AVAILABLE = False

from interfaces.base_interfaces import LLMProvider
from models.data_models import ModelMetrics
from utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for provider classes
T = TypeVar('T', bound=LLMProvider)

class LangChainOllamaProvider(LLMProvider):
    """
    LangChain-based LLM provider using Ollama for local model inference.
    
    This implements the LLMProvider interface while using LangChain and Ollama
    under the hood.
    """
    
    def __init__(self, 
                 model_name: str = "codellama:instruct",
                 temperature: float = 0.2,
                 max_tokens: int = 2000,
                 top_p: float = 0.9,
                 num_ctx: int = 4096,
                 base_url: str = "http://localhost:11434",
                 num_gpu: int = 0,
                 num_thread: int = os.cpu_count() or 8,
                 timeout: int = 300,
                 keep_alive: str = "10m"):
        """
        Initialize the LangChain Ollama provider.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0-1)
            num_ctx: Context window size
            base_url: Base URL for the Ollama server
            num_gpu: Number of GPU layers to use (0 for CPU-only)
            num_thread: Number of CPU threads to use
            timeout: Request timeout in seconds
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain dependencies not available. "
                "Please install with: pip install langchain-community"
            )
            
        self.model_name = model_name
        self.temperature = max(0.0, min(2.0, temperature))  # Clamp 0-2
        self.max_tokens = max(1, max_tokens)  # Ensure at least 1
        self.top_p = max(0.0, min(1.0, top_p))  # Clamp 0-1
        self.num_ctx = max(1, num_ctx)  # Ensure at least 1
        self.base_url = base_url.rstrip('/')
        self.num_gpu = max(0, num_gpu)  # Ensure non-negative
        self.num_thread = max(1, num_thread)  # Ensure at least 1
        self.timeout = max(1, timeout)  # Ensure at least 1 second
        self.keep_alive = keep_alive
        
        # Initialize the LangChain LLM
        self._initialize_llm()
        
        logger.info(
            f"Initialized LangChainOllamaProvider with model: {model_name} "
            f"(temperature={temperature}, max_tokens={max_tokens})"
        )
    
    def _initialize_llm(self):
        """
        Initialize the LangChain LLM with Ollama.
        
        Raises:
            RuntimeError: If the LLM fails to initialize
        """
        try:
            # Log initialization attempt
            logger.info(
                f"Initializing Ollama LLM with model: {self.model_name} "
                f"at {self.base_url}"
            )
            
            # Setup callback manager for streaming
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Initialize Ollama LLM with all configured parameters
            self.llm = Ollama(
                base_url=self.base_url,
                model=self.model_name,
                callback_manager=callback_manager,
                temperature=self.temperature,
                top_p=self.top_p,
                num_ctx=self.num_ctx,
                num_gpu=self.num_gpu,
                num_thread=self.num_thread,
                timeout=self.timeout,
                keep_alive=self.keep_alive,
            )
            
            # Verify the model is available by making a test call
            try:
                # Just check if the model responds, don't process the output
                self.llm.invoke("Test connection")
            except Exception as test_error:
                logger.warning(
                    f"Test call to Ollama model failed: {test_error}"
                    "The model might not be properly loaded or the server might be unreachable."
                )
                # Don't fail here, as the model might still work for actual requests
                
            logger.info(f"Successfully initialized Ollama LLM: {self.model_name}")
            
        except Exception as e:
            error_msg = f"Failed to initialize Ollama LLM: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Provide more helpful error messages for common issues
            if "ConnectionError" in str(e):
                error_msg = (
                    f"Failed to connect to Ollama server at {self.base_url}. "
                    "Please ensure the Ollama server is running and accessible."
                )
            elif "404" in str(e) and "model" in str(e).lower():
                error_msg = (
                    f"Model '{self.model_name}' not found. "
                    f"Please ensure the model is downloaded with: ollama pull {self.model_name}"
                )
                
            raise RuntimeError(error_msg) from e
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the configured Ollama model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters that will override defaults
                    (e.g., temperature, max_tokens, top_p)
            
        Returns:
            The generated text as a string
            
        Raises:
            RuntimeError: If text generation fails for any reason
        """
        logger.debug(f"Starting text generation with LangChainOllamaProvider")
        logger.debug(f"Model: {self.model_name}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        logger.debug(f"Generation parameters: {kwargs}")
        if not prompt or not isinstance(prompt, str):
            logger.warning("Empty or invalid prompt provided")
            return ""
            
        try:
            # Log the generation request (without logging the full prompt for privacy)
            logger.debug(
                f"Generating text with model: {self.model_name}, "
                f"temperature: {kwargs.get('temperature', self.temperature)}"
            )
            
            # Prepare generation parameters, allowing overrides from kwargs
            gen_params = {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'top_p': self.top_p,
                'num_ctx': self.num_ctx,
                'timeout': kwargs.get('timeout', self.timeout),
                **kwargs  # Allow overrides from kwargs
            }
            
            # Generate the response
            start_time = time.time()
            response = self.llm.invoke(prompt, **gen_params)
            response_time = time.time() - start_time
            
            # Log the successful generation
            logger.debug(
                f"Generated {len(response)} characters in {response_time:.2f}s "
                f"(model: {self.model_name})"
            )
            
            # Clean and return the response
            return response.strip()
            
        except Exception as e:
            error_msg = f"Error generating text with model {self.model_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Provide more specific error messages for common issues
            if "context length" in str(e).lower():
                error_msg = (
                    f"Prompt too long for model context window (max {self.num_ctx} tokens). "
                    "Please reduce the prompt length or increase context window size."
                )
            elif "timeout" in str(e).lower():
                error_msg = (
                    "Request timed out. The model may be under heavy load or the prompt is too complex. "
                    f"Current timeout: {self.timeout}s"
                )
                
            raise RuntimeError(error_msg) from e
    
    def is_available(self) -> bool:
        """
        Check if the LLM service and model are available and responding.
        
        Returns:
            bool: True if the service is available and the model can be used,
                  False otherwise
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain dependencies not available")
            return False
            
        if not hasattr(self, 'llm') or self.llm is None:
            logger.warning("LLM not initialized")
            return False
            
        try:
            # Try a simple request with a short timeout to check availability
            test_prompt = "Test availability - please respond with 'OK'"
            
            # Use a shorter timeout for the availability check
            response = self.llm.invoke(
                test_prompt,
                max_tokens=10,  # Only need a short response
                temperature=0,  # Use deterministic output
                timeout=10      # Shorter timeout for availability check
            )
            
            # If we got here, the model responded
            logger.debug(f"LLM service is available. Model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.warning(
                f"LLM service is not available (model: {self.model_name}): {str(e)}",
                exc_info=logger.isEnabledFor(logging.DEBUG)  # Only log full trace in debug mode
            )
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current model and configuration.
        
        Returns:
            Dictionary containing comprehensive model information including:
            - Provider details
            - Model configuration
            - Performance settings
            - System information
        """
        import platform
        import sys
        
        # Basic model info
        model_info = {
            # Provider information
            'provider': 'LangChain Ollama',
            'provider_version': '1.0.0',
            
            # Model configuration
            'model_name': self.model_name,
            'model_architecture': 'CodeLLaMA',  # Default, will be updated if available
            'context_window': self.num_ctx,
            
            # Generation parameters
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'timeout': self.timeout,
            
            # Hardware configuration
            'num_gpu_layers': self.num_gpu,
            'num_threads': self.num_thread,
            
            # Server information
            'server_url': self.base_url,
            'server_status': 'available' if self.is_available() else 'unavailable',
            
            # System information
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'langchain_available': LANGCHAIN_AVAILABLE
        }
        
        # Try to get more detailed model information if available
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                # Add any additional model-specific information
                model_info.update({
                    'model_type': getattr(self.llm, 'model_type', 'unknown'),
                    'model_family': getattr(self.llm, 'model_family', 'unknown'),
                })
            except Exception as e:
                logger.debug(f"Could not get additional model info: {e}")
        
        return model_info
    
    def get_metrics(self) -> ModelMetrics:
        """
        Get comprehensive metrics about the LLM usage and performance.
        
        This includes both basic metrics from the parent class and additional
        LangChain/Ollama specific metrics.
        
        Returns:
            ModelMetrics: Object containing detailed usage and performance metrics
        """
        # Start with the base metrics
        metrics = super().get_metrics()
        
        # Add LangChain/Ollama specific metrics if available
        if hasattr(self, '_metrics'):
            # Convert our internal metrics to the standard format
            metrics.total_requests = self._metrics.get('total_requests', 0)
            metrics.successful_requests = self._metrics.get('successful_requests', 0)
            metrics.failed_requests = self._metrics.get('failed_requests', 0)
            metrics.total_tokens_generated = self._metrics.get('total_tokens_generated', 0)
            metrics.total_tokens_processed = self._metrics.get('total_tokens_processed', 0)
            
            # Add timing information if available
            if 'request_timings' in self._metrics and self._metrics['request_timings']:
                timings = self._metrics['request_timings']
                
                # Update metrics using the ModelMetrics methods
                for response_time in timings:
                    metrics.update_metrics(
                        response_time=response_time,
                        success=True,  # Assuming all recorded timings are for successful requests
                        tokens_used=metrics.total_tokens_processed / len(timings) if metrics.total_tokens_processed > 0 else None
                    )
                
                # Keep only recent timings (last 100 requests) to prevent memory issues
                if len(timings) > 100:
                    self._metrics['request_timings'] = timings[-100:]
        
        # Add model-specific information
        model_info = self.get_model_info()
        metrics.metadata.update({
            'model_name': model_info.get('model_name', 'unknown'),
            'model_architecture': model_info.get('model_architecture', 'unknown'),
            'context_window': model_info.get('context_window', 0),
            'server_status': model_info.get('server_status', 'unknown'),
            'provider': 'LangChainOllama',
            'langchain_version': '0.1.0',  # Should be dynamically determined in production
            'ollama_server': model_info.get('server_url', 'unknown')
        })
        
        return metrics
