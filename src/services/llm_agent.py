from typing import Dict, Any, List, Optional
import requests
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from core.prompt_builder import build_generation_prompt
from src.config.settings import config as app_config

logger = logging.getLogger(__name__)

@dataclass
class CodeLlamaConfig:
    base_url: str = app_config.langchain.ollama.base_url
    model: str = app_config.model.llm_model_name or app_config.langchain.ollama.model_name
    temperature: float = app_config.model.llm_temperature or app_config.langchain.ollama.temperature
    max_tokens: int = app_config.langchain.ollama.max_tokens
    top_p: float = app_config.model.llm_top_p or app_config.langchain.ollama.top_p

class CodeLlamaAgent:
    def __init__(self, config: Optional[CodeLlamaConfig] = None):
        self.config = config or CodeLlamaConfig()
        self.conversation_history: List[Dict[str, str]] = []

    def generate_code(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate code using the local CodeLlama instance."""
        url = f"{self.config.base_url}/api/generate"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
                "top_p": self.config.top_p,
            }
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Update conversation history
            self.conversation_history.append({"role": "assistant", "content": result["response"]})
            
            return result["response"]
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []

    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        """Extract code blocks from markdown text."""
        import re
        pattern = r"```(?:kotlin)?\n(.*?)\n```"
        return re.findall(pattern, text, re.DOTALL)

class TestGeneratorAgent:
    def __init__(self, llm_agent: CodeLlamaAgent):
        self.llm_agent = llm_agent
        # Use centralized prompt templates via core.prompt_builder

    def generate_test(
        self, 
        code: str, 
        test_class_name: str,
        similar_tests: Optional[List[str]] = None
    ) -> str:
        """Generate test code for the given Kotlin code."""
        # Build prompt using the shared PromptBuilder (legacy helper)
        prompt = build_generation_prompt(
            class_name=test_class_name,
            class_code=code,
            similar_tests=similar_tests or []
        )
        
        return self.llm_agent.generate_code(
            prompt=prompt,
            system_prompt="You are a senior Kotlin developer specializing in writing comprehensive unit tests.",
            temperature=0.3
        )

def create_test_generator_agent() -> TestGeneratorAgent:
    """Factory function to create a test generator agent."""
    cfg = CodeLlamaConfig()
    llm_agent = CodeLlamaAgent(cfg)
    return TestGeneratorAgent(llm_agent)
