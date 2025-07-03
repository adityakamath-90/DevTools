# API Documentation
## Kotlin Development Tools - Component APIs

### Overview
This document describes the APIs and interfaces for all components in the Kotlin Development Tools project, with a focus on the enhanced test generation system with semantic similarity matching.

## Core Classes and APIs

### 1. KotlinTestGenerator (Main Component)

**Purpose**: Main orchestrator for AI-powered test case generation with semantic similarity matching.

```python
class KotlinTestGenerator:
    def __init__(self, source_dir: str, test_dir: str, llm_client: LLMClient, indexer: EmbeddingIndexer)
    def extract_class_name(self, code: str) -> Optional[str]
    def clean_generated_code(self, generated_code: str) -> str
    def process_file(self, filepath: str) -> None
    def generate_tests_for_all(self) -> None
```

#### Constructor Parameters
- `source_dir` (str): Directory containing Kotlin source files
- `test_dir` (str): Output directory for generated test files
- `llm_client` (LLMClient): Interface to the LLM service
- `indexer` (EmbeddingIndexer): Semantic similarity indexer

#### Methods

##### `extract_class_name(code: str) -> Optional[str]`
Enhanced class name extraction with intelligent prioritization.

**Features:**
- Removes comments to avoid false matches
- Distinguishes between data classes and regular classes
- Prioritizes main classes over data classes
- Handles multiple class definitions in a single file

**Parameters:**
- `code` (str): Kotlin source code content

**Returns:**
- `Optional[str]`: Extracted class name or None if not found

##### `clean_generated_code(generated_code: str) -> str`
Removes markdown formatting from generated test code.

**Parameters:**
- `generated_code` (str): Raw generated code with potential markdown

**Returns:**
- `str`: Clean Kotlin code without markdown formatting

##### `process_file(filepath: str) -> None`
Processes a single Kotlin file and generates corresponding test cases.

**Workflow:**
1. Extract class name from source code
2. Retrieve similar test cases using embeddings
3. Generate test code with AI using context
4. Generate accuracy feedback
5. Clean and save the generated test file

##### `generate_tests_for_all() -> None`
Batch processes all Kotlin files in the source directory.

---

### 2. EmbeddingIndexer

**Purpose**: Manages semantic indexing and similarity search for existing test cases using sentence transformers and FAISS.

```python
class EmbeddingIndexer:
    def __init__(self, test_dir: str)
    def load_test_cases(self) -> List[str]
    def build_index(self) -> None
    def retrieve_similar(self, query: str, top_k: int = 3) -> List[str]
```

#### Enhanced Features
- Uses `sentence-transformers` with `all-MiniLM-L6-v2` model
- FAISS index for fast similarity search
- Graceful handling of empty test case directories
- Top-K similarity matching with configurable results

#### Methods

##### `retrieve_similar(query: str, top_k: int = 3) -> List[str]`
Finds the most similar existing test cases for a given source code.

**Parameters:**
- `query` (str): Source code to find similar tests for
- `top_k` (int): Number of similar tests to return (default: 3)

**Returns:**
- `List[str]`: List of similar test case contents

---

### 3. LLMClient

**Purpose**: Provides a unified interface for interacting with Large Language Models via Ollama.

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
