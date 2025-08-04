"""
LangChain service for local AI-powered test generation and validation.

This module provides integration with LangChain and Ollama for generating and validating
test cases using locally running language models.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from models.data_models import TestCase, GenerationResult, GenerationStatus
from interfaces.base_interfaces import TestGenerator, SimilarityIndexer
from utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class LocalLLMConfig:
    """Configuration for local LLM using Ollama."""
    model_name: str = "codellama:7b"  # Default to CodeLLaMA 7B
    temperature: float = 0.2
    max_tokens: int = 2000
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 2048  # Context window size
    num_gpu: int = 1  # Number of GPU layers to use (-1 for all)
    num_thread: int = 4  # Number of CPU threads to use
    timeout: int = 300  # Timeout in seconds for model responses
    base_url: str = "http://localhost:11434"  # Ollama server URL
    embedding_model: str = "codellama:7b"  # Model to use for embeddings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for Ollama API."""
        return {
            "model": self.model_name,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
            },
            "stream": False
        }

@dataclass
class TestGenerationConfig:
    """Configuration for test generation."""
    llm_config: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    max_retries: int = 3
    batch_size: int = 4  # For processing multiple examples
    max_improvement_attempts: int = 3  # Max attempts to improve a test
    min_quality_score: float = 0.7  # Minimum quality score to accept a test

class LocalTestGenerator(TestGenerator):
    """
    Local test generator using Ollama and LangChain for test generation and validation.
    """
    
    def __init__(self, 
                 llm_client: Any = None,
                 embedding_model: Optional[Embeddings] = None,
                 config: Optional[TestGenerationConfig] = None):
        """
        Initialize the local test generator.
        
        Args:
            llm_client: Optional pre-initialized LLM client
            embedding_model: Optional pre-initialized embedding model
            config: Configuration for test generation
        """
        self.config = config or TestGenerationConfig()
        self.llm = llm_client or self._initialize_llm()
        self.embedding_model = embedding_model or self._initialize_embeddings()
        self.vector_store = None
        
        # Initialize prompt templates
        self.test_generation_prompt = self._create_test_generation_prompt()
        self.test_validation_prompt = self._create_test_validation_prompt()
        self.test_improvement_prompt = self._create_test_improvement_prompt()
        
        # Initialize output parsers
        self.json_parser = JsonOutputParser()
        
        logger.info(f"Initialized LocalTestGenerator with model: {self.config.llm_config.model_name}")
    
    def _initialize_llm(self):
        """Initialize the local LLM using Ollama."""
        try:
            # Setup callback manager for streaming
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Initialize Ollama LLM
            return Ollama(
                base_url=self.config.llm_config.base_url,
                model=self.config.llm_config.model_name,
                callback_manager=callback_manager,
                temperature=self.config.llm_config.temperature,
                top_p=self.config.llm_config.top_p,
                top_k=self.config.llm_config.top_k,
                num_ctx=self.config.llm_config.num_ctx,
                num_gpu=self.config.llm_config.num_gpu,
                num_thread=self.config.llm_config.num_thread,
                repeat_penalty=self.config.llm_config.repeat_penalty,
                timeout=self.config.llm_config.timeout
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
            raise RuntimeError("Failed to initialize local LLM. Make sure Ollama is running and the model is downloaded.")
    
    def _initialize_embeddings(self):
        """Initialize the local embedding model using Ollama."""
        try:
            return OllamaEmbeddings(
                base_url=self.config.llm_config.base_url,
                model=self.config.llm_config.embedding_model,
                num_gpu=self.config.llm_config.num_gpu,
                num_thread=self.config.llm_config.num_thread
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise RuntimeError("Failed to initialize embedding model. Make sure the embedding model is available in Ollama.")
    
    def _create_test_generation_prompt(self) -> PromptTemplate:
        """Create a prompt template for test generation optimized for local models."""
        template = """<s>[INST] <<SYS>>
You are an expert Kotlin developer. Generate a comprehensive JUnit 5 test class for the provided Kotlin class.

Follow these guidelines:
1. Test all public methods with appropriate test cases
2. Include edge cases and error conditions
3. Use proper assertions and test organization
4. Follow Kotlin testing best practices
5. Use descriptive test method names
6. Include necessary imports
7. Add comments for complex test cases
<</SYS>>

Kotlin class to test:
```kotlin
{class_code}
```

Additional context:
{additional_context}

Generate a complete JUnit 5 test class. Return ONLY the test class code, without any explanations or markdown formatting. [/INST]\n```kotlin
"""
        return PromptTemplate.from_template(template)
    
    def _create_test_validation_prompt(self) -> PromptTemplate:
        """Create a prompt template for test validation."""
        template = """<s>[INST] <<SYS>>
You are a senior software engineer reviewing a Kotlin test case. Analyze the test case and provide structured feedback.

Your task is to evaluate the test case based on:
1. Compilation and runtime errors
2. Test coverage of the original class
3. Edge cases and error conditions
4. Test quality and maintainability
5. Performance considerations

Return your response as a JSON object with the specified structure.
<</SYS>>

Original class:
```kotlin
{class_code}
```

Generated test:
```kotlin
{test_code}
```

Compilation/Test errors (if any):
{errors}

Provide your analysis in the following JSON format:
{{
    "is_valid": boolean,
    "feedback": "Detailed feedback on the test case",
    "suggested_fixes": ["list", "of", "suggested", "fixes"],
    "test_coverage_score": number (0-100),
    "test_quality_score": number (0-100)
}}

Return ONLY the JSON object, no other text or markdown. [/INST]\n```json
"""
        return PromptTemplate.from_template(template)
    
    def _create_test_improvement_prompt(self) -> PromptTemplate:
        """Create a prompt template for test improvement."""
        template = """<s>[INST] <<SYS>>
You are a test improvement expert. Your task is to improve the provided test case based on the feedback.

Guidelines:
1. Address all issues mentioned in the feedback
2. Maintain or improve test coverage
3. Ensure all tests are valid and follow best practices
4. Keep the code clean and well-organized
5. Add comments for complex test cases

Return ONLY the improved test code, without any explanations or markdown formatting.
<</SYS>>

Original class:
```kotlin
{class_code}
```

Current test:
```kotlin
{test_code}
```

Feedback to address:
{feedback}

Generate the improved test case: [/INST]\n```kotlin
"""
        return PromptTemplate.from_template(template)
    
    def generate_test(self, class_code: str, class_name: str, additional_context: str = "") -> GenerationResult:
        """
        Generate a test for the given class code.
        
        Args:
            class_code: The source code of the class to test
            class_name: The name of the class
            additional_context: Any additional context or requirements
            
        Returns:
            GenerationResult containing the generated test and metadata
        """
        try:
            # Create the test generation chain
            chain = (
                {
                    "class_code": lambda x: x["class_code"],
                    "class_name": lambda x: x["class_name"],
                    "additional_context": lambda x: x["additional_context"]
                }
                | self.test_generation_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate the test
            test_code = chain.invoke({
                "class_code": class_code,
                "class_name": class_name,
                "additional_context": additional_context
            })
            
            return GenerationResult(
                status=GenerationStatus.SUCCESS,
                test_cases=[TestCase(name=f"Generated test for {class_name}", code=test_code)],
                metrics={"generation_attempts": 1}
            )
            
        except Exception as e:
            logger.error(f"Error generating test for {class_name}: {str(e)}")
            return GenerationResult(
                status=GenerationStatus.FAILED,
                error_message=str(e),
                metrics={"error": str(e)}
            )
    
    def validate_test(self, class_code: str, test_code: str, errors: str = "") -> Dict[str, Any]:
        """
        Validate a generated test against the original class.
        
        Args:
            class_code: The source code of the original class
            test_code: The generated test code to validate
            errors: Any compilation or runtime errors from the test
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Create the validation chain
            chain = (
                {
                    "class_code": lambda x: x["class_code"],
                    "test_code": lambda x: x["test_code"],
                    "errors": lambda x: x["errors"]
                }
                | self.test_validation_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Get validation feedback
            feedback = chain.invoke({
                "class_code": class_code,
                "test_code": test_code,
                "errors": errors
            })
            
            # Parse the JSON response
            import json
            return json.loads(feedback.strip())
            
        except Exception as e:
            logger.error(f"Error validating test: {str(e)}")
            return {
                "is_valid": False,
                "feedback": f"Error during validation: {str(e)}",
                "suggested_fixes": [],
                "test_coverage_score": 0,
                "test_quality_score": 0
            }
    
    def improve_test(self, class_code: str, test_code: str, feedback: str) -> str:
        """
        Improve a test based on feedback.
        
        Args:
            class_code: The source code of the original class
            test_code: The current test code to improve
            feedback: Feedback on how to improve the test
            
        Returns:
            Improved test code
        """
        try:
            # Create the improvement chain
            chain = (
                {
                    "class_code": lambda x: x["class_code"],
                    "test_code": lambda x: x["test_code"],
                    "feedback": lambda x: x["feedback"]
                }
                | self.test_improvement_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate improved test
            improved_test = chain.invoke({
                "class_code": class_code,
                "test_code": test_code,
                "feedback": feedback
            })
            
            return improved_test.strip()
            
        except Exception as e:
            logger.error(f"Error improving test: {str(e)}")
            return test_code  # Return original if improvement fails

    def generate_and_validate_test(self, class_code: str, class_name: str, max_attempts: int = 3) -> GenerationResult:
        """
        Generate and validate a test with automatic improvement iterations.
        
        Args:
            class_code: The source code of the class to test
            class_name: The name of the class
            max_attempts: Maximum number of improvement attempts
            
        Returns:
            GenerationResult with the best test and validation metrics
        """
        attempts = 0
        best_test = None
        best_score = 0
        validation_results = []
        
        while attempts < max_attempts:
            # Generate or improve test
            if attempts == 0:
                result = self.generate_test(class_code, class_name)
                if result.status != GenerationStatus.SUCCESS:
                    return result
                test_code = result.test_cases[0].code
            else:
                feedback = validation_results[-1]["feedback"]
                test_code = self.improve_test(class_code, test_code, feedback)
            
            # Validate the test
            validation = self.validate_test(class_code, test_code)
            validation_results.append(validation)
            
            # Check if this is the best test so far
            current_score = validation.get("test_quality_score", 0)
            if current_score > best_score:
                best_score = current_score
                best_test = test_code
            
            # If test is valid, we're done
            if validation.get("is_valid", False) and current_score >= 80:  # 80% threshold
                break
                
            attempts += 1
        
        # Prepare the final result
        metrics = {
            "validation_attempts": attempts + 1,
            "final_quality_score": best_score,
            "validation_results": validation_results
        }
        
        return GenerationResult(
            status=GenerationStatus.SUCCESS,
            test_cases=[TestCase(name=f"Validated test for {class_name}", code=best_test)],
            metrics=metrics
        )
