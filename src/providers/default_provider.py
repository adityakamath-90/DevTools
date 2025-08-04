"""
Default LLM Provider implementation.

This module provides a default implementation of the LLMProvider interface
that maintains the original functionality as a fallback when other providers
are not available.
"""

import time
import json
import requests
from typing import Dict, Any, Optional

from interfaces.base_interfaces import LLMProvider
from models.data_models import ModelMetrics
from config.settings import LLMConfig
from utils.logging import get_logger

logger = get_logger(__name__)

class DefaultLLMProvider(LLMProvider):
    """
    Default LLM provider that maintains the original implementation.
    
    This is used as a fallback when other providers are not available.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the default LLM provider.
        
        Args:
            config: Configuration for the LLM service
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.metrics = ModelMetrics(model_name=config.model_name)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the default LLM provider.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        try:
            # Prepare the request payload
            payload = self._prepare_payload(prompt, **kwargs)
            
            # Make the API request
            response = requests.post(
                self.config.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout
            )
            
            # Process the response
            if response.status_code == 200:
                logger.info(f"Response status: {response.status_code}")
                
                # Handle streaming response (multiple JSON objects, one per line)
                response_text = ""
                try:
                    # Process each line as a separate JSON object
                    for line in response.iter_lines():
                        if line:  # filter out keep-alive new lines
                            try:
                                chunk = json.loads(line)
                                if 'response' in chunk:
                                    response_text += chunk['response']
                                logger.debug(f"Received chunk: {chunk}")
                            except json.JSONDecodeError as je:
                                logger.warning(f"Failed to parse chunk: {line}")
                    
                    logger.debug(f"Final combined response text: {response_text}")
                    self._update_metrics(time.time() - start_time, success=True)
                    return response_text
                    
                except Exception as e:
                    logger.error(f"Error processing streaming response: {str(e)}")
                    raise
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                self._update_metrics(time.time() - start_time, success=False)
                return ""
                
        except Exception as e:
            self.logger.error(f"Error in default LLM provider: {str(e)}")
            self._update_metrics(time.time() - start_time, success=False)
            return ""
    
    def is_available(self) -> bool:
        """Check if the default provider is available."""
        try:
            response = requests.get(f"{self.config.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "provider": "default",
            "model_name": self.config.model_name,
            "api_url": self.config.api_url
        }
    
    def get_metrics(self) -> ModelMetrics:
        """Get performance metrics."""
        return self.metrics
    
    def _prepare_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare the request payload."""
        return {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "model": self.config.model_name
        }
    
    def _extract_text(self, response_data: Dict[str, Any]) -> str:
        """Extract the generated text from the API response."""
        # Handle Ollama API response format
        if "response" in response_data:
            return response_data["response"]
        # Fallback to OpenAI-compatible format
        elif "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0].get("text", "")
        # If no valid response format found, log the response for debugging
        self.logger.warning(f"Unexpected response format: {response_data}")
        return ""
    
    def _update_metrics(self, response_time: float, success: bool) -> None:
        """Update performance metrics."""
        self.metrics.update_metrics(
            response_time=response_time,
            success=success
        )
