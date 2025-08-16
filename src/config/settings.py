"""
Configuration management for the AI-powered Kotlin test generation system.
Handles environment variables, model settings, and system parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

from .langchain_config import LangChainConfig, default_config as default_langchain_config


@dataclass
class ModelConfig:
    """Configuration for AI models and embedding systems."""
    
    # LLM Configuration
    ollama_api_url: str = "http://127.0.0.1:11434/api/generate"
    llm_model_name: str = "codellama:instruct"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.8
    
    # LangChain Configuration
    use_langchain: bool = True
    langchain_provider: str = "langchain_ollama"
    
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
    
    input_dir: str = "input-src"
    output_dir: str = "output-test"
    reference_tests_dir: str = "testcase-datastore"
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


@dataclass
class Config:
    """Main configuration class that aggregates all configuration objects."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.directories = DirectoryConfig()
        self.processing = ProcessingConfig()
        self.langchain = default_langchain_config
        self._load_from_environment()
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        
        # LangChain Configuration
        if os.getenv("USE_LANGCHAIN") is not None:
            self.model.use_langchain = os.getenv("USE_LANGCHAIN").lower() in ("true", "1", "t")
        if os.getenv("LANGCHAIN_PROVIDER"):
            self.model.langchain_provider = os.getenv("LANGCHAIN_PROVIDER")
        
        # If NOT using LangChain, honor legacy direct LLM env vars; otherwise, ignore them to prevent conflicts
        if not self.model.use_langchain:
            if os.getenv("OLLAMA_API_URL"):
                self.model.ollama_api_url = os.getenv("OLLAMA_API_URL")
            if os.getenv("LLM_MODEL_NAME"):
                self.model.llm_model_name = os.getenv("LLM_MODEL_NAME")
            if os.getenv("LLM_TEMPERATURE"):
                self.model.llm_temperature = float(os.getenv("LLM_TEMPERATURE"))
            if os.getenv("LLM_TOP_P"):
                self.model.llm_top_p = float(os.getenv("LLM_TOP_P"))
        
        # Update LangChain config from environment
        if os.getenv("LANGCHAIN_MODEL_NAME"):
            self.langchain.ollama.model_name = os.getenv("LANGCHAIN_MODEL_NAME")
        if os.getenv("LANGCHAIN_TEMPERATURE"):
            self.langchain.ollama.temperature = float(os.getenv("LANGCHAIN_TEMPERATURE"))
        if os.getenv("LANGCHAIN_MAX_TOKENS"):
            self.langchain.ollama.max_tokens = int(os.getenv("LANGCHAIN_MAX_TOKENS"))
        if os.getenv("LANGCHAIN_TOP_P"):
            self.langchain.ollama.top_p = float(os.getenv("LANGCHAIN_TOP_P"))
        if os.getenv("LANGCHAIN_BASE_URL"):
            self.langchain.ollama.base_url = os.getenv("LANGCHAIN_BASE_URL")
        if os.getenv("LANGCHAIN_NUM_CTX"):
            self.langchain.ollama.num_ctx = int(os.getenv("LANGCHAIN_NUM_CTX"))
        if os.getenv("LANGCHAIN_NUM_GPU"):
            self.langchain.ollama.num_gpu = int(os.getenv("LANGCHAIN_NUM_GPU"))
        if os.getenv("LANGCHAIN_NUM_THREAD"):
            self.langchain.ollama.num_thread = int(os.getenv("LANGCHAIN_NUM_THREAD"))
        if os.getenv("LANGCHAIN_TIMEOUT"):
            self.langchain.ollama.timeout = int(os.getenv("LANGCHAIN_TIMEOUT"))
        if os.getenv("LANGCHAIN_KEEP_ALIVE"):
            self.langchain.ollama.keep_alive = os.getenv("LANGCHAIN_KEEP_ALIVE")
        if os.getenv("LANGCHAIN_ENABLE_STREAMING_CALLBACKS"):
            self.langchain.ollama.enable_streaming_callbacks = os.getenv("LANGCHAIN_ENABLE_STREAMING_CALLBACKS").lower() in ("true", "1", "t")
        if os.getenv("LANGCHAIN_WARMUP_ON_START"):
            self.langchain.ollama.warmup_on_start = os.getenv("LANGCHAIN_WARMUP_ON_START").lower() in ("true", "1", "t")
        if os.getenv("LANGCHAIN_WARMUP_MAX_TOKENS"):
            self.langchain.ollama.warmup_max_tokens = int(os.getenv("LANGCHAIN_WARMUP_MAX_TOKENS"))
        if os.getenv("LANGCHAIN_WARMUP_PROMPT"):
            self.langchain.ollama.warmup_prompt = os.getenv("LANGCHAIN_WARMUP_PROMPT")
        if os.getenv("LANGCHAIN_AUTO_TUNE_HARDWARE"):
            self.langchain.ollama.auto_tune_hardware = os.getenv("LANGCHAIN_AUTO_TUNE_HARDWARE").lower() in ("true", "1", "t")
        
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
    """Configuration for test generation."""
    # Required directories
    source_dir: str = "input-src"
    output_dir: str = "output-test"
    existing_tests_dir: str = "testcase-datastore"
    
    # Generation parameters
    max_files: int = 0  # 0 means no limit
    enable_validation: bool = True
    min_coverage: float = 80.0
    classpath: Optional[str] = None
    
    # Performance optimization flags (all disabled by default for speed)
    enable_compilation_checks: bool = False
    enable_static_analysis: bool = True
    enable_auto_fix: bool = False
    enable_coverage_checks: bool = True
    enable_coverage_improvement: bool = True
    max_similar_tests: int = 1
    similarity_top_k: int = 3  # Number of similar tests to retrieve
    max_source_code_chars: int = 2000
    max_similar_context_chars: int = 2500
    
    # LLM generation parameters for speed
    temperature: float = 0.25
    max_tokens: int = 500
    top_p: float = 0.9
    num_ctx: int = 2048
    timeout: int = 45  # seconds
    
    def __post_init__(self):
        """Ensure directories exist and load optional overrides from environment."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.existing_tests_dir).mkdir(parents=True, exist_ok=True)
        
        # Optional env overrides for prompt truncation and retrieval
        if v := os.getenv("MAX_SOURCE_CODE_CHARS"):
            try:
                self.max_source_code_chars = int(v)
            except ValueError:
                pass
        if v := os.getenv("MAX_SIMILAR_CONTEXT_CHARS"):
            try:
                self.max_similar_context_chars = int(v)
            except ValueError:
                pass
        if v := os.getenv("MAX_SIMILAR_TESTS"):
            try:
                self.max_similar_tests = int(v)
            except ValueError:
                pass
        if v := os.getenv("SIMILARITY_TOP_K"):
            try:
                self.similarity_top_k = int(v)
            except ValueError:
                pass
        
        # Optional env overrides for generation behavior
        if v := os.getenv("GEN_TEMPERATURE"):
            try:
                self.temperature = float(v)
            except ValueError:
                pass
        if v := os.getenv("GEN_MAX_TOKENS"):
            try:
                self.max_tokens = int(v)
            except ValueError:
                pass
        if v := os.getenv("GEN_TOP_P"):
            try:
                self.top_p = float(v)
            except ValueError:
                pass
        if v := os.getenv("GEN_NUM_CTX"):
            try:
                self.num_ctx = int(v)
            except ValueError:
                pass
        if v := os.getenv("GEN_TIMEOUT"):
            try:
                self.timeout = int(v)
            except ValueError:
                pass
        
        # Optional toggles
        if v := os.getenv("ENABLE_VALIDATION"):
            self.enable_validation = v.lower() in ("true", "1", "t", "yes")
        if v := os.getenv("ENABLE_STATIC_ANALYSIS"):
            self.enable_static_analysis = v.lower() in ("true", "1", "t", "yes")
        if v := os.getenv("ENABLE_COVERAGE_CHECKS"):
            self.enable_coverage_checks = v.lower() in ("true", "1", "t", "yes")
        if v := os.getenv("ENABLE_COVERAGE_IMPROVEMENT"):
            self.enable_coverage_improvement = v.lower() in ("true", "1", "t", "yes")


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
    test_cases_dir: str = "testcase-datastore" 
    max_length: int = 512
    batch_size: int = 8
    # Centralized flags
    allow_model_download: bool = False
    offline_mode: bool = True
    set_env_offline: bool = True
    
    def __post_init__(self):
        """Load from environment variables."""
        self.model_name = os.getenv("EMBEDDING_MODEL", self.model_name)
        
        if batch_size := os.getenv("EMBEDDING_BATCH_SIZE"):
            self.batch_size = int(batch_size)
        if max_length := os.getenv("EMBEDDING_MAX_LENGTH"):
            self.max_length = int(max_length)
        # Flags
        if allow_dl := os.getenv("ALLOW_MODEL_DOWNLOAD"):
            self.allow_model_download = allow_dl.lower() in ("true", "1", "yes")
        if offline := os.getenv("OFFLINE_MODE"):
            self.offline_mode = offline.lower() in ("true", "1", "yes")
        if set_env := os.getenv("SET_ENV_OFFLINE"):
            self.set_env_offline = set_env.lower() in ("true", "1", "yes")
