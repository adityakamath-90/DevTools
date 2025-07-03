# Migration Guide: New Modular Architecture

## Overview

The AI-powered Kotlin test generation system has been refactored into a modern, modular architecture following industry best practices. This guide explains the changes and how to migrate from the legacy system.

## What's New

### 🏗️ Modular Architecture
- **Clean separation of concerns** with distinct layers
- **Industry-standard project structure** with proper module organization
- **Configuration management** using dataclasses and environment variables
- **Comprehensive error handling** and logging throughout
- **Interface-driven design** for better testability and extensibility

### 📁 New Directory Structure
```
src/
├── config/          # Configuration management
├── models/          # Data models and schemas
├── interfaces/      # Abstract interfaces and protocols
├── core/           # Core business logic
├── services/       # Service implementations
└── utils/          # Utility functions and helpers
```

### ✨ Key Improvements
1. **Robust Error Handling**: Comprehensive exception handling and graceful fallbacks
2. **Detailed Logging**: Structured logging with component-based loggers
3. **Configuration System**: Environment-based configuration with sensible defaults
4. **Metrics and Monitoring**: Built-in performance tracking and health checks
5. **Backward Compatibility**: Legacy interfaces maintained for smooth migration
6. **Type Safety**: Comprehensive type hints and data validation

## Migration Options

### Option 1: Continue Using Legacy Interface (Recommended for Quick Start)

The legacy scripts still work exactly as before:

```bash
# Test generation (works as before)
python TestCaseGenerator.py

# KDoc generation (works as before)  
python KdocGenerator.py
```

**No changes required!** Your existing workflows will continue to work.

### Option 2: Use New Modular Interface (Recommended for New Development)

Use the new `main.py` script for better features:

```bash
# Test generation with new interface
python main.py test --source-dir src/input-src --output-dir output-test

# KDoc generation with new interface
python main.py kdoc --source-dir src/input-src

# Health check
python main.py health

# Performance metrics
python main.py metrics
```

### Option 3: Programmatic Usage (Recommended for Integration)

Use the new modular components in your own code:

```python
from src.config.settings import GenerationConfig
from src.services.llm_service import LLMService
from src.services.embedding_service import EmbeddingIndexerService
from src.core.test_generator import KotlinTestGenerator

# Configure the system
config = GenerationConfig(
    source_dir="path/to/kotlin/files",
    output_dir="path/to/output"
)

# Initialize services
llm_service = LLMService()
embedding_service = EmbeddingIndexerService()

# Create test generator
generator = KotlinTestGenerator(
    llm_provider=llm_service,
    similarity_indexer=embedding_service,
    config=config
)

# Generate tests
results = generator.generate_tests_for_directory("src/input-src")
```

## Key Benefits of Migration

### 1. Better Error Handling
- **Before**: Silent failures or cryptic error messages
- **After**: Detailed error reporting with context and suggestions

### 2. Improved Configuration
- **Before**: Hardcoded values scattered throughout code
- **After**: Centralized configuration with environment variable support

### 3. Enhanced Logging
- **Before**: Print statements throughout code
- **After**: Structured logging with levels and component identification

### 4. Performance Monitoring
- **Before**: No visibility into performance
- **After**: Built-in metrics tracking and health checks

### 5. Better Testability
- **Before**: Monolithic classes difficult to test
- **After**: Interface-driven design with dependency injection

## Configuration

### Environment Variables
Set these to customize behavior:

```bash
# LLM Configuration
export OLLAMA_API_URL="http://127.0.0.1:11434/api/generate"
export MODEL_NAME="codellama:instruct"

# Embedding Configuration  
export EMBEDDING_MODEL="microsoft/codebert-base"

# Generation Settings
export MAX_TOKENS=2048
export TEMPERATURE=0.0
export SIMILARITY_TOP_K=3
```

### Programmatic Configuration
```python
from src.config.settings import GenerationConfig, LLMConfig

# Custom configuration
config = GenerationConfig(
    source_dir="custom/source/path",
    output_dir="custom/output/path",
    enable_validation=True,
    similarity_top_k=5
)

llm_config = LLMConfig(
    model_name="custom-model",
    temperature=0.2,
    max_tokens=4096
)
```

## Advanced Features

### 1. Health Monitoring
```bash
python main.py health
```
Output:
```
LLM Service: ✅ Available
Embedding Service: ✅ Available  
KDoc Service: ✅ Available
Overall Health: ✅ Healthy
```

### 2. Performance Metrics
```bash
python main.py metrics
```
Output:
```
Performance Metrics:

Test Generator:
  total_requests: 25
  successful_requests: 23
  success_rate: 92.0
  average_response_time: 2.34

LLM Service:
  api_url: http://127.0.0.1:11434/api/generate
  model_name: codellama:instruct
  is_available: True
```

### 3. Batch Processing
```python
# Process multiple directories
directories = ["src/main", "src/test", "src/examples"]
for dir_path in directories:
    results = generator.generate_tests_for_directory(dir_path)
    print(f"Processed {dir_path}: {len(results)} files")
```

### 4. Custom Service Integration
```python
# Use your own LLM service
class CustomLLMService(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your custom implementation
        return custom_generate(prompt)

# Use custom service
generator = KotlinTestGenerator(
    llm_provider=CustomLLMService(),
    similarity_indexer=embedding_service,
    config=config
)
```

## Migration Timeline

### Phase 1: Evaluation (Current)
- Test the new system alongside the legacy system
- Verify that legacy interfaces still work
- Run health checks and metrics

### Phase 2: Gradual Migration (Recommended)
- Start using `main.py` for new workflows
- Gradually migrate scripts to use new interfaces
- Update documentation and training materials

### Phase 3: Full Migration (Future)
- Deprecate legacy scripts
- Use only new modular interfaces
- Take advantage of advanced features

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src.config'
   ```
   **Solution**: Run from the project root directory or use the provided scripts

2. **Service Unavailable**
   ```
   LLM Service: ❌ Unavailable
   ```
   **Solution**: Ensure Ollama server is running: `ollama serve`

3. **Missing Dependencies**
   ```
   ImportError: No module named 'transformers'
   ```
   **Solution**: Install dependencies: `pip install -r requirements.txt`

### Getting Help

1. **Check Health**: `python main.py health`
2. **Review Logs**: Enable debug mode with `--debug` flag
3. **Check Configuration**: Verify environment variables
4. **Fallback Mode**: Use legacy scripts if new system has issues

## Best Practices

### 1. Use Configuration Files
Create environment-specific configs:
```python
# config/development.py
from src.config.settings import GenerationConfig

config = GenerationConfig(
    source_dir="src/input-src",
    output_dir="output-test", 
    enable_validation=True,
    debug_mode=True
)
```

### 2. Implement Custom Services
Extend the system with your own services:
```python
class CustomEmbeddingService(SimilarityIndexer):
    def find_similar(self, query: str, top_k: int = 3) -> List[str]:
        # Your custom similarity logic
        return custom_find_similar(query, top_k)
```

### 3. Monitor Performance
Regularly check system metrics:
```python
metrics = generator.get_metrics()
if metrics.success_rate < 80:
    logger.warning("Low success rate detected")
```

### 4. Use Structured Logging
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing file", extra={"file_path": file_path})
```

## Support

For questions or issues with the migration:

1. **Check the documentation** in the `docs/` directory
2. **Review the examples** in `main.py` and legacy scripts
3. **Test with health checks** before reporting issues
4. **Provide logs and metrics** when seeking help

The new architecture provides a solid foundation for future enhancements while maintaining full backward compatibility with existing workflows.
