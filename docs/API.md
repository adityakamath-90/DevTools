# API Documentation
## AI-Powered Kotlin Test Generation System v2.0 - Component APIs

### Overview
This document describes the APIs and interfaces for all components in the AI-powered Kotlin test generation system v2.0, featuring a modular architecture with service-oriented design, interface-driven development, and comprehensive configuration management.

## New Modular Architecture APIs

### 1. Configuration Management API

**Purpose**: Provides flexible, environment-based configuration management for all system components.

#### GenerationConfig (src/config/settings.py)

```python
@dataclass
class GenerationConfig:
    source_dir: str = "input-src"
    test_dir: str = "output-test"
    test_datastore_dir: str = "testcase-datastore"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'GenerationConfig'
    def override_from_env(self) -> 'GenerationConfig'
```

#### LLMConfig (src/config/settings.py)

```python
@dataclass  
class LLMConfig:
    api_url: str = "http://127.0.0.1:11434/api/generate"
    model_name: str = "codellama:instruct"
    timeout: int = 300
    max_retries: int = 3
    temperature: float = 0.1
```

#### EmbeddingConfig (src/config/settings.py)

```python
@dataclass
class EmbeddingConfig:
    model_name: str = "microsoft/codebert-base"
    batch_size: int = 8
    max_length: int = 512
    use_gpu: bool = True
```

### 2. Service Layer APIs

**Purpose**: Provides clean abstractions for all external dependencies and AI services.

#### LLMService (src/services/llm_service.py)

```python
class LLMService:
    def __init__(self, config: LLMConfig)
    def generate_code(self, prompt: str) -> str
    def health_check() -> bool
    def get_model_info() -> dict
```

**Key Methods:**

##### `generate_code(prompt: str) -> str`
Generates code using the configured LLM with robust error handling.

**Features:**
- Automatic retry on failure
- Timeout handling
- Comprehensive error logging
- Response validation and cleaning

**Parameters:**
- `prompt` (str): The prompt to send to the LLM

**Returns:**
- `str`: Generated code response

**Example:**
```python
llm_service = LLMService(LLMConfig())
prompt = "Generate a JUnit 5 test for Calculator class"
generated_code = llm_service.generate_code(prompt)
```

##### `health_check() -> bool`
Checks if the LLM service is available and responding.

**Returns:**
- `bool`: True if service is healthy, False otherwise

#### EmbeddingIndexerService (src/services/embedding_service.py)

```python
class EmbeddingIndexerService:
    def __init__(self, config: EmbeddingConfig)
    def index_files(self, file_patterns: List[str]) -> None
    def find_similar_content(self, query: str, top_k: int = 5) -> List[str]
    def health_check() -> bool
```

**Key Methods:**

##### `index_files(file_patterns: List[str]) -> None`
Indexes files for semantic similarity search using Microsoft CodeBERT.

**Parameters:**
- `file_patterns` (List[str]): List of file patterns to index

**Example:**
```python
embedding_service = EmbeddingIndexerService(EmbeddingConfig())
embedding_service.index_files(["src/testcase--datastore/*.kt"])
```

##### `find_similar_content(query: str, top_k: int = 5) -> List[str]`
Finds similar content using semantic similarity.

**Parameters:**
- `query` (str): The query string to find similar content for
- `top_k` (int): Number of similar items to return

**Returns:**
- `List[str]`: List of similar content strings

#### SimpleEmbeddingIndexerService (src/services/embedding_service.py)

```python
class SimpleEmbeddingIndexerService:
    def __init__(self, config: EmbeddingConfig)
    def index_files(self, file_patterns: List[str]) -> None
    def find_similar_content(self, query: str, top_k: int = 5) -> List[str]
    def health_check() -> bool
```

**Purpose**: Lightweight fallback service for environments without ML dependencies.

#### KDocService (src/services/kdoc_service.py)

```python
class KDocService:
    def __init__(self, llm_service: LLMService, config: GenerationConfig)
    def generate_kdoc(self, kotlin_code: str) -> str
    def process_file(self, file_path: str) -> KDocResult
    def process_directory(self, directory_path: str) -> List[KDocResult]
```

### 3. Core Business Logic APIs

**Purpose**: Contains the main business logic separated from infrastructure concerns.

#### KotlinTestGenerator (src/core/test_generator.py)

```python
class KotlinTestGenerator:
    def __init__(self, config: GenerationConfig, llm_service: LLMService, 
                 embedding_service: BaseEmbeddingIndexer)
    def extract_class_name(self, code: str) -> Optional[str]
    def clean_generated_code(self, generated_code: str) -> str
    def process_file(self, filepath: str) -> TestGenerationResult
    def generate_tests_for_all(self) -> List[TestGenerationResult]
```

#### CodeParser (src/core/code_parser.py)

```python
class CodeParser:
    def __init__(self, config: GenerationConfig)
    def parse_kotlin_file(self, file_path: str) -> ParsedKotlinFile
    def extract_classes(self, code: str) -> List[KotlinClass]
    def extract_functions(self, code: str) -> List[KotlinFunction]
    def remove_comments(self, code: str) -> str
```

#### PromptBuilder (src/core/prompt_builder.py)

```python
class PromptBuilder:
    def __init__(self, config: GenerationConfig)
    def build_test_prompt(self, class_code: str, similar_tests: List[str]) -> str
    def build_kdoc_prompt(self, kotlin_code: str) -> str
    def build_context_prompt(self, context: dict) -> str
```

### 4. Interface System APIs

**Purpose**: Provides consistent abstractions and data models.

#### BaseEmbeddingIndexer (src/interfaces/base_interfaces.py)

```python
class BaseEmbeddingIndexer(ABC):
    @abstractmethod
    def index_files(self, file_patterns: List[str]) -> None
    
    @abstractmethod
    def find_similar_content(self, query: str, top_k: int = 5) -> List[str]
    
    @abstractmethod
    def health_check(self) -> bool
```

#### BaseLLMClient (src/interfaces/base_interfaces.py)

```python
class BaseLLMClient(ABC):
    @abstractmethod
    def generate_code(self, prompt: str) -> str
    
    @abstractmethod
    def health_check(self) -> bool
```

#### BaseGenerator (src/interfaces/base_interfaces.py)

```python
class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, input_data: str) -> str
    
    @abstractmethod
    def process_file(self, file_path: str) -> Any
```

### 5. Data Models APIs

**Purpose**: Structured data models for results and communication between components.

#### TestGenerationResult (src/models/data_models.py)

```python
@dataclass
class TestGenerationResult:
    success: bool
    input_file: str
    output_file: str
    class_name: Optional[str] = None
    generated_code: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    similar_tests_found: int = 0
```

#### KDocResult (src/models/data_models.py)

```python
@dataclass
class KDocResult:
    success: bool
    file_path: str
    classes_processed: int = 0
    functions_processed: int = 0
    error_message: Optional[str] = None
    generated_kdoc: Optional[str] = None
```

#### EmbeddingResult (src/models/data_models.py)

```python
@dataclass
class EmbeddingResult:
    query: str
    similar_content: List[str]
    confidence_scores: List[float]
    processing_time: float
```

### 6. Utility APIs

**Purpose**: Shared infrastructure and cross-cutting concerns.

#### Logging System (src/utils/logging.py)

```python
def get_logger(name: str) -> logging.Logger
def setup_logging(level: str = "INFO") -> None
def log_function_call(func_name: str, args: dict) -> None
def log_error(error: Exception, context: dict = None) -> None
```

### 7. CLI Interface API

**Purpose**: Unified command-line interface with health checks and debugging.

#### Main CLI (main.py)

```python
class GenAIApplication:
    def __init__(self, config: Optional[GenerationConfig] = None)
    def health_check(self) -> bool
    def generate_tests(self, source_dir: str = None) -> List[TestGenerationResult]
    def generate_kdoc(self, source_dir: str = None) -> List[KDocResult]
    def generate_all(self, source_dir: str = None) -> dict
```

### 8. Legacy Compatibility APIs

**Purpose**: Backward compatibility with existing scripts.

#### Legacy Test Generator (TestCaseGenerator.py)

```python
# Wrapper that maintains old interface but uses new modular system
def main():
    # Legacy main function that delegates to new system
    pass
```

#### Legacy KDoc Generator (KdocGenerator.py)

```python  
# Wrapper that maintains old interface but uses new modular system
def main():
    # Legacy main function that delegates to new system
    pass
```
4. Generate test code with AI using contextual prompts
5. Generate accuracy validation feedback
6. Clean and format the generated test code
7. Save test file to output directory with backup

**Error Handling:**
- File read/write errors with graceful recovery
- Missing class detection with informative warnings
- LLM generation failures with fallback behavior
- Backup restoration on write failures

**Example:**
```python
generator.process_file("src/input-src/Calculator.kt")
# Generates: output-test/CalculatorTest.kt
```

##### `generate_tests_for_all() -> None`
Batch processes all Kotlin files in the source directory with progress tracking.

**Features:**
- Recursive directory scanning for `.kt` files
- Excludes `testcase--datastore` directory to prevent recursion
- Progress logging and error reporting
- Continues processing on individual file failures

---

### 2. EmbeddingIndexer (Advanced Semantic Similarity)

**Purpose**: Manages semantic indexing and similarity search using Microsoft CodeBERT embeddings and FAISS for high-quality test case matching.

```python
class EmbeddingIndexer:
    def __init__(self, test_dir: str, embedding_model_name: str = "microsoft/codebert-base")
    def _load_and_index(self) -> None
    def _encode(self, texts: List[str]) -> torch.Tensor
    def retrieve_similar(self, code: str, top_k: int = 3) -> List[str]
```

#### Advanced Features
- Uses Microsoft CodeBERT for code-aware embeddings
- FAISS IndexFlatL2 for efficient similarity search
- PyTorch backend for tensor operations
- Automatic model downloading and caching
- Graceful handling of empty test case directories

#### Methods

##### `__init__(test_dir: str, embedding_model_name: str = "microsoft/codebert-base")`
Initializes the embedding indexer with automatic model loading and indexing.

**Parameters:**
- `test_dir` (str): Directory containing reference test cases
- `embedding_model_name` (str): Hugging Face model identifier

**Process:**
1. Downloads and loads CodeBERT model and tokenizer
2. Loads all `.kt` files from test directory
3. Generates embeddings for all test cases
4. Builds FAISS index for similarity search

##### `retrieve_similar(code: str, top_k: int = 3) -> List[str]`
Finds the most semantically similar existing test cases for given source code.

**Parameters:**
- `code` (str): Kotlin source code to find similar tests for
- `top_k` (int): Number of similar tests to return (default: 3)

**Returns:**
- `List[str]`: List of similar test case contents, ranked by similarity

**Process:**
1. Encode input code using CodeBERT
2. Search FAISS index for nearest neighbors
3. Return top-K most similar test cases

**Example:**
```python
indexer = EmbeddingIndexer("src/testcase--datastore/")
similar_tests = indexer.retrieve_similar(kotlin_code, top_k=3)
# Returns: List of 3 most similar test case strings
```

---

### 3. SimpleEmbeddingIndexer (Fallback System)

**Purpose**: Provides a lightweight fallback when advanced ML dependencies are unavailable.

```python
class SimpleEmbeddingIndexer:
    def __init__(self, test_dir: str)
    def _load_test_cases(self) -> None
    def retrieve_similar(self, code: str, top_k: int = 3) -> List[str]
```

#### Features
- No ML dependencies required
- Simple text-based matching
- Fast initialization and operation
- Automatic fallback when EmbeddingIndexer fails

#### Methods

##### `retrieve_similar(code: str, top_k: int = 3) -> List[str]`
Returns available test cases (simplified matching algorithm).

**Parameters:**
- `code` (str): Input code (used for interface compatibility)
- `top_k` (int): Number of test cases to return

**Returns:**
- `List[str]`: First K available test cases

---

### 4. LLMClient (Ollama Integration)

**Purpose**: Provides a unified interface for interacting with CodeLlama via Ollama server.

```python
class LLMClient:
    def __init__(self, api_url: str, model_name: str)
    def generate(self, prompt: str) -> str
```

#### Configuration
- **Default API URL**: `http://127.0.0.1:11434/api/generate`
- **Default Model**: `codellama:instruct`
- **Streaming**: Disabled for reliable response handling

#### Methods

##### `generate(prompt: str) -> str`
Sends a prompt to the LLM and returns the generated response.

**Parameters:**
- `prompt` (str): The input prompt for code generation

**Returns:**
- `str`: Generated text response from the model

**Raises:**
- `requests.RequestException`: If API request fails
- `json.JSONDecodeError`: If response parsing fails

**Example:**
```python
client = LLMClient("http://127.0.0.1:11434/api/generate", "codellama:instruct")
response = client.generate("Generate comprehensive JUnit tests for this Kotlin class...")
```

---

### 4. PromptBuilder

**Purpose**: Constructs intelligent prompts for test generation and accuracy checking with context injection.
class EmbeddingIndexer:
    def __init__(self, test_dir: str, embedding_model_name: str = "microsoft/codebert-base")
    def retrieve_similar(self, code: str, top_k: int = 3) -> List[str]
```

#### Constructor Parameters
- `test_dir` (str): Directory containing existing test cases for indexing
- `embedding_model_name` (str): HuggingFace model identifier for embeddings

#### Attributes
- `test_cases` (List[str]): Loaded test case contents
- `embeddings` (torch.Tensor): Generated embeddings matrix
- `index` (faiss.IndexFlatL2): FAISS similarity index
- `dimension` (int): Embedding vector dimension

#### Methods

##### `retrieve_similar(code: str, top_k: int = 3) -> List[str]`
Finds the most similar existing test cases for given code.

**Parameters:**
- `code` (str): Source code to find similar tests for
- `top_k` (int): Number of similar cases to return (default: 3)

**Returns:**
- `List[str]`: List of similar test case contents

**Example:**
```python
indexer = EmbeddingIndexer("testcase--datastore")
similar_tests = indexer.retrieve_similar(kotlin_code, top_k=5)
```

##### `_encode(texts: List[str]) -> torch.Tensor` (Private)
Generates embeddings for a list of text inputs.

**Parameters:**
- `texts` (List[str]): List of text strings to encode

**Returns:**
- `torch.Tensor`: Embedding matrix of shape (len(texts), embedding_dim)

---

### 3. PromptBuilder

**Purpose**: Constructs context-aware prompts for different generation tasks.

```python
class PromptBuilder:
    @staticmethod
    def build_generation_prompt(class_name: str, class_code: str, similar_tests: List[str]) -> str
    
    @staticmethod
    def generate_accurate_prompt(class_code: str, generated_test: str) -> str
```

#### Static Methods

##### `build_generation_prompt(class_name: str, class_code: str, similar_tests: List[str]) -> str`
Creates a prompt for initial test generation.

**Parameters:**
- `class_name` (str): Name of the Kotlin class being tested
- `class_code` (str): Full source code of the class
- `similar_tests` (List[str]): Relevant existing test cases for context

**Returns:**
- `str`: Formatted prompt for test generation

**Example:**
```python
prompt = PromptBuilder.build_generation_prompt(
    "UserManager", 
    kotlin_class_code, 
    similar_test_cases
)
```

##### `generate_accurate_prompt(class_code: str, generated_test: str) -> str`
Creates a prompt for reviewing and improving generated tests.

**Parameters:**
- `class_code` (str): Original class source code
- `generated_test` (str): Initially generated test code

**Returns:**
- `str`: Formatted prompt for test improvement

---

### 4. KotlinTestGenerator

**Purpose**: Orchestrates the complete test generation workflow.

```python
class KotlinTestGenerator:
    def __init__(self, source_dir: str, test_dir: str, llm_client: LLMClient, indexer: EmbeddingIndexer)
    def process_file(self, filepath: str)
    def generate_tests_for_all(self)
    def extract_class_name(self, code: str) -> Optional[str]
```

#### Constructor Parameters
- `source_dir` (str): Directory containing Kotlin source files
- `test_dir` (str): Output directory for generated test files
- `llm_client` (LLMClient): Configured LLM client instance
- `indexer` (EmbeddingIndexer): Configured embedding indexer instance

#### Methods

##### `generate_tests_for_all()`
Processes all Kotlin files in the source directory and generates tests.

**Side Effects:**
- Creates test files in the configured test directory
- Prints progress information to stdout

**Example:**
```python
generator = KotlinTestGenerator(
    source_dir="input-src",
    test_dir="output-test",
    llm_client=client,
    indexer=indexer
)
generator.generate_tests_for_all()
```

##### `process_file(filepath: str)`
Processes a single Kotlin file and generates its test.

**Parameters:**
- `filepath` (str): Absolute path to the Kotlin source file

**Side Effects:**
- Creates corresponding test file
- Updates internal datastore

##### `extract_class_name(code: str) -> Optional[str]`
Extracts the primary class name from Kotlin source code.

**Parameters:**
- `code` (str): Kotlin source code content

**Returns:**
- `Optional[str]`: Class name if found, None otherwise

---

### 5. KDoc Generation Functions

**Purpose**: Standalone functions for KDoc generation.

#### `generate_kdoc_for_file(file_content: str) -> str`
Generates KDoc comments for a Kotlin file.

**Parameters:**
- `file_content` (str): Complete Kotlin source file content

**Returns:**
- `str`: Enhanced file content with KDoc comments

**Example:**
```python
enhanced_code = generate_kdoc_for_file(kotlin_file_content)
```

#### `update_kdocs_in_file(filepath: str)`
Updates a single file with generated KDoc comments.

**Parameters:**
- `filepath` (str): Path to the Kotlin file to update

**Side Effects:**
- Modifies the file in-place with KDoc comments

#### `update_kdocs_in_directory(directory: str)`
Recursively processes all .kt files in a directory.

**Parameters:**
- `directory` (str): Root directory to process

**Side Effects:**
- Updates all Kotlin files in the directory tree

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_API_URL` | Ollama API endpoint | `http://127.0.0.1:11434/api/generate` |
| `OLLAMA_MODEL` | Model name for generation | `codellama:instruct` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `microsoft/codebert-base` |
| `BATCH_SIZE` | Processing batch size | `10` |
| `MAX_CONTEXT_LENGTH` | Maximum prompt length | `4096` |

### Model Configuration

#### LLM Parameters
```python
{
    "model": "codellama:instruct",
    "temperature": 0,      # Deterministic output
    "top_p": 0.8,         # Nucleus sampling
    "stream": false,      # Blocking request
    "max_tokens": 2048    # Maximum response length
}
```

#### Embedding Parameters
```python
{
    "model_name": "microsoft/codebert-base",
    "max_length": 512,           # Maximum sequence length
    "padding": True,             # Pad sequences
    "truncation": True,          # Truncate long sequences
    "return_tensors": "pt"       # Return PyTorch tensors
}
```

---

## Error Handling

### Exception Types

#### `LLMConnectionError`
Raised when unable to connect to Ollama API.

#### `EmbeddingModelError`
Raised when embedding model fails to load or process.

#### `FileProcessingError`
Raised when file I/O operations fail.

### Error Recovery Strategies

1. **API Failures**: Retry with exponential backoff
2. **Model Loading**: Fallback to alternative models
3. **File Errors**: Skip problematic files and continue
4. **Memory Issues**: Process files in smaller batches

---

## Performance Considerations

### Optimization Guidelines

1. **Batch Processing**: Process multiple files together
2. **Embedding Caching**: Cache embeddings to disk
3. **Model Reuse**: Keep models loaded in memory
4. **Index Persistence**: Save FAISS indices for reuse

### Memory Management

- **Model Loading**: Load models once, reuse across files
- **Embedding Storage**: Use float16 for reduced memory usage
- **Batch Size**: Adjust based on available system memory
- **Index Size**: Monitor FAISS index memory consumption

### Scalability Limits

| Component | Limit | Recommendation |
|-----------|-------|----------------|
| File Size | 50KB per file | Split large files |
| Batch Size | 100 files | Process in chunks |
| Index Size | 10K test cases | Implement hierarchical indexing |
| Memory Usage | 8GB RAM | Use GPU acceleration |

---

## Integration Examples

### Basic KDoc Generation
```python
# Simple KDoc generation
from src.Kdoc import update_kdocs_in_directory

# Process entire project
update_kdocs_in_directory("/path/to/kotlin/project/src")
```

### Advanced Test Generation
```python
# Full test generation pipeline
from src.TestCaseGenerator import KotlinTestGenerator
from src.LLMClient import LLMClient
from src.EmbeddingIndexer import EmbeddingIndexer

# Initialize components
llm_client = LLMClient("http://127.0.0.1:11434/api/generate", "codellama:instruct")
indexer = EmbeddingIndexer("testcase--datastore")

# Create generator
generator = KotlinTestGenerator(
    source_dir="src/main/kotlin",
    test_dir="src/test/kotlin",
    llm_client=llm_client,
    indexer=indexer
)

# Generate tests
generator.generate_tests_for_all()
```

### Custom Pipeline
```python
# Custom processing pipeline
import os
from src.LLMClient import LLMClient
from src.PromptBuilder import PromptBuilder

def custom_kdoc_generation(file_path: str):
    """Custom KDoc generation with additional validation"""
    
    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Build custom prompt
    prompt = f"Generate comprehensive KDoc for:\n{content}"
    
    # Generate with LLM
    client = LLMClient("http://127.0.0.1:11434/api/generate", "codellama:instruct")
    result = client.generate(prompt)
    
    # Validate result
    if "/**" in result and "*/" in result:
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(result)
        return True
    
    return False
```

---

## Testing the APIs

### Unit Test Examples

```python
import unittest
from unittest.mock import Mock, patch
from src.LLMClient import LLMClient

class TestLLMClient(unittest.TestCase):
    
    def setUp(self):
        self.client = LLMClient("http://test-url", "test-model")
    
    @patch('requests.post')
    def test_generate_success(self, mock_post):
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Generated code"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.client.generate("Test prompt")
        self.assertEqual(result, "Generated code")
    
    @patch('requests.post')
    def test_generate_failure(self, mock_post):
        # Mock API failure
        mock_post.side_effect = Exception("Connection failed")
        
        result = self.client.generate("Test prompt")
        self.assertEqual(result, "")  # Should return empty string on error
```

### Integration Test Examples

```python
import tempfile
import os
from src.TestCaseGenerator import KotlinTestGenerator

def test_end_to_end_generation():
    """Test complete test generation workflow"""
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = os.path.join(temp_dir, "src")
        test_dir = os.path.join(temp_dir, "test")
        os.makedirs(source_dir)
        
        # Create sample Kotlin file
        kotlin_code = """
        class Calculator {
            fun add(a: Int, b: Int): Int = a + b
            fun subtract(a: Int, b: Int): Int = a - b
        }
        """
        
        with open(os.path.join(source_dir, "Calculator.kt"), 'w') as f:
            f.write(kotlin_code)
        
        # Initialize generator (would need mocked components for testing)
        # ... test implementation
```
