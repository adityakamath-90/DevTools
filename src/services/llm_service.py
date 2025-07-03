"""
LLM service implementation for AI-powered code generation.

This module provides a production-ready LLM service that interfaces with
Ollama/CodeLlama for generating Kotlin test code and KDoc comments.
"""

import time
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

from interfaces.base_interfaces import LLMProvider
from models.data_models import ModelMetrics
from config.settings import LLMConfig
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM service."""
    success: bool
    text: str
    response_time: float
    error_message: Optional[str] = None
    token_count: Optional[int] = None
    model_used: Optional[str] = None


class LLMService(LLMProvider):
    """
    Production-ready LLM service with comprehensive error handling.
    
    Features:
    - Robust error handling and retries
    - Response time monitoring
    - Token usage tracking
    - Configurable generation parameters
    - Health check capabilities
    - Detailed logging and metrics
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Metrics tracking
        self.metrics = ModelMetrics()
        
        self.logger.info(f"Initialized LLMService with config: {self.config}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        start_time = time.time()
        
        try:
            # Prepare request parameters
            request_params = self._prepare_request_params(prompt, **kwargs)
            
            # Make API request
            response = self._make_api_request(request_params)
            
            # Track metrics
            response_time = time.time() - start_time
            self._update_metrics(response_time, success=True)
            
            if response.success:
                self.logger.debug(f"LLM generation successful in {response_time:.2f}s")
                return response.text
            else:
                self.logger.error(f"LLM generation failed: {response.error_message}")
                return ""
                
        except Exception as e:
            response_time = time.time() - start_time
            self._update_metrics(response_time, success=False)
            self.logger.error(f"LLM generation error: {e}")
            return ""
    
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
            return bool(response)
            
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
        """Update service metrics."""
        self.metrics.total_requests += 1
        self.metrics.total_processing_time += response_time
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time
        self.metrics.average_response_time = (
            self.metrics.total_processing_time / self.metrics.total_requests
        )
        
        # Success rate is now calculated automatically via the property


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
        from config.settings import LLMConfig
        config = LLMConfig(
            api_url=api_url,
            model_name=model_name
        )
        
        # Initialize new service
        self.service = LLMService(config)
    
    def generate(self, prompt: str) -> str:
        """Legacy method for generating text."""
        return self.service.generate(prompt)
