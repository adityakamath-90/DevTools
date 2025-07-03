"""
Interface definitions for the AI-powered Kotlin test generation system.
Defines abstract base classes and protocols for consistent implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Protocol
from models.data_models import (
    KotlinClass, 
    GenerationRequest, 
    GenerationResult, 
    SimilarityMatch,
    EmbeddingVector,
    ModelMetrics
)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def encode(self, texts: List[str]) -> List[EmbeddingVector]:
        """Encode texts into embedding vectors."""
        ...
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        ...


class SimilarityIndexer(ABC):
    """Abstract base class for similarity indexing systems."""
    
    @abstractmethod
    def build_index(self, embeddings: List[EmbeddingVector]) -> None:
        """Build similarity search index from embeddings."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: EmbeddingVector, top_k: int = 3) -> List[SimilarityMatch]:
        """Search for similar items in the index."""
        pass
    
    @abstractmethod
    def add_to_index(self, embeddings: List[EmbeddingVector]) -> None:
        """Add new embeddings to existing index."""
        pass
    
    @property
    @abstractmethod
    def index_size(self) -> int:
        """Get the number of items in the index."""
        pass


class LLMProvider(ABC):
    """Abstract base class for Large Language Model providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the given prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass


class TestGenerator(ABC):
    """Abstract base class for test generators."""
    
    @abstractmethod
    def generate_tests(self, request: GenerationRequest) -> GenerationResult:
        """Generate tests for a given request."""
        pass
    
    @abstractmethod
    def extract_class_info(self, source_code: str, file_path: str) -> Optional[KotlinClass]:
        """Extract class information from source code."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> ModelMetrics:
        """Get performance metrics."""
        pass


class CodeParser(ABC):
    """Abstract base class for code parsers."""
    
    @abstractmethod
    def parse_kotlin_file(self, file_path: str) -> List[KotlinClass]:
        """Parse a Kotlin file and extract class information."""
        pass
    
    @abstractmethod
    def extract_methods(self, class_code: str) -> List[str]:
        """Extract method signatures from class code."""
        pass
    
    @abstractmethod
    def extract_properties(self, class_code: str) -> List[str]:
        """Extract property definitions from class code."""
        pass
    
    @abstractmethod
    def is_data_class(self, class_code: str) -> bool:
        """Check if a class is a data class."""
        pass


class FileProcessor(ABC):
    """Abstract base class for file processors."""
    
    @abstractmethod
    def process_file(self, file_path: str) -> GenerationResult:
        """Process a single file."""
        pass
    
    @abstractmethod
    def process_directory(self, directory_path: str) -> List[GenerationResult]:
        """Process all files in a directory."""
        pass
    
    @abstractmethod
    def create_backup(self, file_path: str) -> str:
        """Create a backup of a file."""
        pass
    
    @abstractmethod
    def restore_backup(self, backup_path: str, original_path: str) -> bool:
        """Restore a file from backup."""
        pass


class PromptBuilder(ABC):
    """Abstract base class for prompt builders."""
    
    @abstractmethod
    def build_generation_prompt(
        self, 
        kotlin_class: KotlinClass, 
        similar_tests: List[str]
    ) -> str:
        """Build a prompt for test generation."""
        pass
    
    @abstractmethod
    def build_validation_prompt(
        self, 
        kotlin_class: KotlinClass, 
        generated_test: str
    ) -> str:
        """Build a prompt for test validation."""
        pass
    
    @abstractmethod
    def build_improvement_prompt(
        self, 
        kotlin_class: KotlinClass, 
        test_code: str, 
        feedback: str
    ) -> str:
        """Build a prompt for test improvement."""
        pass


class CacheProvider(ABC):
    """Abstract base class for caching systems."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class MetricsCollector(ABC):
    """Abstract base class for metrics collection."""
    
    @abstractmethod
    def record_generation_request(self, request: GenerationRequest) -> None:
        """Record a test generation request."""
        pass
    
    @abstractmethod
    def record_generation_result(self, result: GenerationResult) -> None:
        """Record a test generation result."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> ModelMetrics:
        """Get collected metrics."""
        pass
    
    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        pass


class Logger(Protocol):
    """Protocol for logging."""
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        ...
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        ...
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        ...
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        ...


class ConfigProvider(Protocol):
    """Protocol for configuration providers."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...
    
    def validate(self) -> bool:
        """Validate configuration."""
        ...


# Type aliases for common interfaces
EmbeddingIndexer = SimilarityIndexer  # Alias for backward compatibility
