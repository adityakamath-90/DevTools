"""
Configuration management for the AI-powered Kotlin test generation system.
Handles environment variables, model settings, and system parameters.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for AI models and embedding systems."""
    
    # LLM Configuration
    ollama_api_url: str = "http://127.0.0.1:11434/api/generate"
    llm_model_name: str = "codellama:instruct"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.8
    
    # Embedding Configuration
    embedding_model_name: str = "microsoft/codebert-base"
    embedding_dimension: Optional[int] = None
    max_sequence_length: int = 512
    batch_size: int = 8
    
    # FAISS Configuration
    faiss_index_type: str = "IndexFlatL2"
    similarity_top_k: int = 3


@dataclass
class DirectoryConfig:
    """Configuration for input/output directories."""
    
    input_dir: str = "src/input-src"
    output_dir: str = "output-test"
    reference_tests_dir: str = "src/testcase--datastore"
    backup_dir: str = "backups"
    docs_dir: str = "docs"
    
    def create_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        for dir_path in [self.output_dir, self.reference_tests_dir, self.backup_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    
    # File processing
    supported_extensions: tuple = (".kt",)
    exclude_patterns: tuple = ("test", "Test", "__pycache__")
    max_file_size_mb: int = 10
    
    # Test generation
    enable_accuracy_feedback: bool = True
    enable_backup_creation: bool = True
    skip_existing_tests: bool = False
    
    # Logging
    log_level: str = "INFO"
    enable_progress_tracking: bool = True


class Config:
    """Main configuration class that aggregates all configuration objects."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.directories = DirectoryConfig()
        self.processing = ProcessingConfig()
        self._load_from_environment()
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        
        # Model configuration from environment
        self.model.ollama_api_url = os.getenv("OLLAMA_API_URL", self.model.ollama_api_url)
        self.model.llm_model_name = os.getenv("MODEL_NAME", self.model.llm_model_name)
        self.model.embedding_model_name = os.getenv("EMBEDDING_MODEL", self.model.embedding_model_name)
        
        # Temperature and other model parameters
        if temp := os.getenv("LLM_TEMPERATURE"):
            self.model.llm_temperature = float(temp)
        if top_p := os.getenv("LLM_TOP_P"):
            self.model.llm_top_p = float(top_p)
        if batch_size := os.getenv("EMBEDDING_BATCH_SIZE"):
            self.model.batch_size = int(batch_size)
        
        # Directory configuration
        self.directories.input_dir = os.getenv("INPUT_DIR", self.directories.input_dir)
        self.directories.output_dir = os.getenv("OUTPUT_DIR", self.directories.output_dir)
        self.directories.reference_tests_dir = os.getenv("REFERENCE_TESTS_DIR", self.directories.reference_tests_dir)
        
        # Processing configuration
        self.processing.log_level = os.getenv("LOG_LEVEL", self.processing.log_level)
        if skip_existing := os.getenv("SKIP_EXISTING_TESTS"):
            self.processing.skip_existing_tests = skip_existing.lower() in ("true", "1", "yes")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            "model": self.model.__dict__,
            "directories": self.directories.__dict__,
            "processing": self.processing.__dict__
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate directory paths
            if not Path(self.directories.input_dir).exists():
                raise ValueError(f"Input directory does not exist: {self.directories.input_dir}")
            
            # Validate model parameters
            if not (0.0 <= self.model.llm_temperature <= 2.0):
                raise ValueError(f"Invalid temperature: {self.model.llm_temperature}")
            
            if not (0.0 <= self.model.llm_top_p <= 1.0):
                raise ValueError(f"Invalid top_p: {self.model.llm_top_p}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Configuration validation failed: {e}")
            return False


# Global configuration instance
config = Config()


# Additional configuration classes for specific components
@dataclass
class GenerationConfig:
    """Configuration for test generation workflow."""
    
    source_dir: str = "input-src"
    output_dir: str = "output-test"
    existing_tests_dir: str = "testcase-datastore"
    
    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 2048
    similarity_top_k: int = 3
    
    # Feature flags
    enable_validation: bool = True
    enable_backup: bool = True
    debug_mode: bool = False
    
    def __post_init__(self):
        """Ensure directories exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.existing_tests_dir).mkdir(parents=True, exist_ok=True)


@dataclass  
class LLMConfig:
    """Configuration for LLM service."""
    
    api_url: str = "http://127.0.0.1:11434/api/generate"
    model_name: str = "codellama:instruct"
    temperature: float = 0.1
    top_p: float = 0.8
    max_tokens: int = 2048
    timeout: int = 300  # 5 minutes
    
    def __post_init__(self):
        """Load from environment variables."""
        self.api_url = os.getenv("OLLAMA_API_URL", self.api_url)
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        
        if temp := os.getenv("LLM_TEMPERATURE"):
            self.temperature = float(temp)
        if top_p := os.getenv("LLM_TOP_P"):
            self.top_p = float(top_p)
        if max_tokens := os.getenv("MAX_TOKENS"):
            self.max_tokens = int(max_tokens)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""
    
    model_name: str = "microsoft/codebert-base"
    test_cases_dir: str = "src/testcase--datastore" 
    max_length: int = 512
    batch_size: int = 8
    
    def __post_init__(self):
        """Load from environment variables."""
        self.model_name = os.getenv("EMBEDDING_MODEL", self.model_name)
        
        if batch_size := os.getenv("EMBEDDING_BATCH_SIZE"):
            self.batch_size = int(batch_size)
        if max_length := os.getenv("EMBEDDING_MAX_LENGTH"):
            self.max_length = int(max_length)
