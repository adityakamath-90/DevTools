# Architecture Documentation
## AI-Powered Kotlin Test Generation System v2.0

### Overview
This project implements an advanced AI-powered system for Kotlin test generation that automatically creates comprehensive JUnit 5 test cases using Large Language Models (CodeLlama) and semantic similarity matching with Microsoft CodeBERT embeddings. The system has been completely redesigned with a modular, production-ready architecture featuring interface-driven design, robust configuration management, and comprehensive logging.

## 🎯 Key Achievements & System Capabilities

### ✅ Production-Ready Features (v2.0)
- **Modular Architecture**: Interface-driven design with dependency injection
- **Service-Oriented Design**: Clear separation of concerns with service layer
- **Configuration Management**: Environment-based configuration with override capabilities
- **Structured Logging**: Comprehensive logging system with configurable levels
- **Error Handling**: Graceful degradation and robust error recovery
- **Backward Compatibility**: Legacy scripts still work alongside new modular system
- **CLI Interface**: Unified command-line interface with health checks and debugging

### ✅ Enhanced AI Features
- **Intelligent Test Generation**: Creates comprehensive JUnit 5 test cases with MockK support
- **Semantic Similarity Matching**: Microsoft CodeBERT embeddings with FAISS indexing
- **Fallback System**: SimpleEmbeddingIndexer for environments without ML dependencies
- **Advanced Class Detection**: Smart regex patterns with comment filtering and class prioritization
- **Clean Code Output**: Automatic markdown removal and production-ready formatting
- **Comprehensive Error Handling**: Graceful degradation with backup and recovery mechanisms
- **Batch Processing**: Handles entire directories with progress tracking and logging

### 🔧 Advanced Technical Features
- **Dual Indexing System**: Advanced CodeBERT embeddings with simple text fallback
- **Context-Aware Generation**: Uses similar existing test patterns for improved quality
- **Intelligent Class Prioritization**: Distinguishes between data classes and regular classes
- **Comment-Aware Parsing**: Removes single-line and multi-line comments before processing
- **Markdown Cleaning Pipeline**: Removes formatting artifacts from generated code
- **Backup and Recovery**: Creates backups before modifications with automatic restoration

### 🧠 ML/AI Integration
- **Microsoft CodeBERT**: Code-aware embeddings for semantic similarity
- **FAISS Vector Search**: Efficient similarity matching with IndexFlatL2
- **CodeLlama Integration**: Via Ollama for high-quality code generation
- **PyTorch Backend**: Tensor operations for embedding computation
- **Hugging Face Integration**: Automatic model downloading and caching

## System Architecture v2.0

### New Modular Architecture

The system now follows a layered, service-oriented architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer                                │
│  main.py (unified CLI) | Legacy Scripts (backward compatibility) │
├─────────────────────────────────────────────────────────────────┤
│                        Core Layer                               │
│  test_generator.py | code_parser.py | prompt_builder.py         │
├─────────────────────────────────────────────────────────────────┤
│                       Service Layer                             │
│  llm_service.py | llm_agent.py | embedding_service.py | kdoc_service.py │
├─────────────────────────────────────────────────────────────────┤
│                    Configuration Layer                          │
│  settings.py (environment-based config with overrides)          │
├─────────────────────────────────────────────────────────────────┤
│                     Foundation Layer                            │
│  interfaces/ | models/ | utils/ (logging, helpers)             │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/
├── config/              # Configuration management
│   ├── __init__.py
│   └── settings.py      # Environment-based configuration
├── core/                # Core business logic
│   ├── __init__.py
│   ├── code_parser.py   # Kotlin code parsing and analysis
│   ├── prompt_builder.py # Context-aware prompt construction
│   └── test_generator.py # Test generation orchestrator
├── services/            # Service layer
│   ├── __init__.py
│   ├── llm_service.py   # LLM interface with error handling (LangChain/requests)
│   ├── llm_agent.py     # Lightweight Ollama CodeLlama agent (no LangChain)
│   ├── embedding_service.py # Semantic similarity service
│   └── kdoc_service.py  # KDoc generation service
├── providers/           # LLM provider integrations
│   ├── __init__.py
│   └── langchain_provider.py # LangChain-based Ollama provider
├── interfaces/          # Abstract base classes
│   ├── __init__.py
│   └── base_interfaces.py # Interface definitions
├── models/              # Data models
│   ├── __init__.py
│   └── data_models.py   # Result objects and data structures
└── utils/               # Utilities
    ├── __init__.py
    └── logging.py       # Structured logging system
```
### High-Level Architecture

For comprehensive visual system architecture diagrams, see **[DIAGRAMS.md](./DIAGRAMS.md)** which contains detailed Mermaid diagrams showing:
- Updated system component relationships with modular architecture
- Enhanced data flow architecture with service layer
- Class interaction diagrams with interface patterns
- Deployment structure with configuration management
- Sequence diagrams showing the complete generation workflow

## Enhanced Component Architecture v2.0

### 1. Configuration Management System ⭐ *New Feature*

The configuration system provides flexible, environment-based configuration management.

**Core Components:**
- **GenerationConfig**: Main configuration dataclass with environment overrides
- **LLMConfig**: LLM service configuration (model, API endpoints, timeouts)
- **EmbeddingConfig**: Embedding service configuration (models, batch sizes)
- **Environment Integration**: Automatic environment variable loading and validation

**Features:**
- Environment variable overrides for all settings
- Dataclass-based configuration with type safety
- Validation and default value handling
- Development vs production configuration profiles

### 2. Service Layer Architecture ⭐ *Core Enhanced Feature*

The service layer provides clean abstractions for all external dependencies and AI services.

**Core Services:**
- **LLMService**: Interface to Ollama/CodeLlama with robust error handling
- **EmbeddingIndexerService**: Advanced semantic indexing using Microsoft CodeBERT
- **SimpleEmbeddingIndexerService**: Lightweight fallback for constrained environments
- **KDocService**: Documentation generation service with template system

**Alternative LLM Paths:**
- **LangChain Provider**: `src/providers/langchain_provider.py` implements `LLMProvider` via LangChain + Ollama.
- **Lightweight Agent**: `src/services/llm_agent.py` (`CodeLlamaAgent`, `TestGeneratorAgent`) uses direct REST to Ollama with minimal state. Useful for low-dependency local setups.

**Enhanced Features:**
- Interface-driven design with dependency injection
- Comprehensive error handling and retry mechanisms
- Automatic fallback between advanced and simple implementations
- Structured logging throughout all services
- Health check capabilities for monitoring

### 3. Core Business Logic ⭐ *Redesigned Architecture*

The core layer contains the main business logic separated from infrastructure concerns.

**Core Components:**
- **KotlinTestGenerator**: Main orchestrator with enhanced error handling
- **CodeParser**: Advanced Kotlin code parsing with comment filtering
- **PromptBuilder**: Context-aware prompt construction with template system

**Enhanced Workflow:**
1. **File Discovery**: Recursive scanning with intelligent filtering
2. **Class Extraction**: Comment-aware parsing with multiple class handling
3. **Semantic Analysis**: Service-based embedding with automatic fallback
4. **Context Building**: Integration of similar test patterns via services
5. **AI Generation**: Service-based LLM interaction with structured prompts
6. **Quality Assurance**: Validation and feedback loops
7. **Code Cleaning**: Markdown removal and formatting standardization
8. **Safe File Operations**: Backup creation and atomic writes

### 4. Interface and Model System ⭐ *New Architecture*

The interface system provides consistent abstractions and data models.

**Core Interfaces:**
- **BaseEmbeddingIndexer**: Abstract base for all embedding services
- **BaseLLMClient**: Abstract base for all LLM integrations
- **BaseGenerator**: Abstract base for all generation services

**Data Models:**
- **TestGenerationResult**: Structured result objects with metadata
- **KDocResult**: Documentation generation results with validation
- **EmbeddingResult**: Semantic similarity results with confidence scores

### 5. Utility and Infrastructure ⭐ *Enhanced System*

The utility layer provides shared infrastructure and cross-cutting concerns.

**Core Utilities:**
- **Structured Logging**: Configurable logging with different levels and formats
- **Error Handling**: Consistent exception handling patterns
- **File Operations**: Safe file I/O with backup and recovery
- **Configuration Loading**: Environment-based configuration with validation

### 6. Legacy Compatibility Layer ⭐ *Backward Compatibility*

The legacy layer provides backward compatibility while encouraging migration.

**Legacy Components:**
- **TestCaseGenerator.py**: Legacy wrapper around new modular system
- **KdocGenerator.py**: Legacy wrapper for documentation generation
- **Migration utilities**: Tools to help transition from legacy to modular system

## Data Flow Architecture v2.0

For detailed data flow visualization, see the comprehensive flow diagrams in **[DIAGRAMS.md](./DIAGRAMS.md)**.

### New Modular Data Flow

```
Input Files → Core Parser → Service Layer → Model Layer → Output Files
                ↓              ↓              ↓
        Configuration → Logging System → Error Handling
                ↓              ↓              ↓
        Health Checks → Monitoring → Fallback Systems
```

### Service Integration Flow

1. **Configuration Loading**: Environment-based settings with validation
2. **Service Initialization**: Dependency injection with health checks
3. **Core Processing**: Business logic with service abstractions
4. **Error Handling**: Graceful degradation with fallback mechanisms
5. **Result Processing**: Structured results with metadata
6. **Output Generation**: Safe file operations with backup/recovery

**Enhanced Test Generation Flow:**
```
1. Kotlin File Input → TestCaseGenerator.process_file()
2. Class Analysis → extract_class_name() [Enhanced regex with comment removal]
3. Embedding Creation → EmbeddingIndexer.retrieve_similar()
4. Semantic Search → FAISS similarity matching with existing tests
5. Context Building → PromptBuilder.build_generation_prompt() [with similar tests]
6. AI Generation → LLMClient.generate() → CodeLlama inference
7. Quality Check → PromptBuilder.generate_accurate_prompt() → Feedback generation
8. Code Cleaning → clean_generated_code() [Remove markdown formatting]
9. File Output → {ClassName}Test.kt in output-test directory
```

**KDoc Generation Flow:**
```
Kotlin File → File Reader → Code Analysis → Prompt Creation → LLM Request → 
CodeLlama → KDoc Response → File Update → Enhanced Kotlin File with KDocs
```

**Embedding Index Flow:**
```
Test Files in testcase--datastore → Sentence Transformer Encoding → 
FAISS Index Construction → Semantic Search Ready
```

## Technical Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **LLM Engine** | Ollama + CodeLlama:instruct | Latest | Code generation and documentation |
| **Embeddings** | sentence-transformers | 2.2.2 | Semantic similarity matching |
| **Embedding Model** | all-MiniLM-L6-v2 | Latest | Text encoding for similarity |
| **Vector Search** | FAISS | 1.7.4 | Fast similarity search and clustering |
| **ML Framework** | PyTorch + Transformers | 4.31.0+ | Model inference and operations |
| **Language** | Python | 3.8+ | Core implementation |

### Dependencies

```python
# Core ML/AI
sentence-transformers==2.2.2
transformers==4.31.0
torch>=1.9.0

# Vector Search
faiss-cpu==1.7.4
numpy==1.24.3

# API & Utilities
requests==2.31.0
huggingface-hub==0.16.4
```

## Module Descriptions

### 1. LLMClient.py
**Purpose**: Unified interface for LLM interactions
- Handles Ollama API communication
- Manages request/response processing
- Implements error handling and retries
- Configures generation parameters

### 2. EmbeddingIndexer.py
**Purpose**: Semantic search and similarity matching
- Loads and indexes existing test cases
- Uses CodeBERT for code embeddings
- Builds FAISS index for fast retrieval
- Provides similarity search functionality

### 3. PromptBuilder.py
**Purpose**: Dynamic prompt construction
- Creates context-aware prompts
- Includes relevant examples
- Specifies generation requirements
- Handles different prompt types (KDoc vs Tests)

### 4. TestCaseGenerator.py
**Purpose**: Orchestrates test generation workflow
- Manages the complete test generation pipeline
- Coordinates between all components
- Handles file I/O operations
- Implements generation strategies

### 5. Kdoc.py / KdocGenerator.py
**Purpose**: Documentation generation
- Processes Kotlin source files
- Generates comprehensive KDoc comments
- Preserves existing documentation
- Updates files in-place

## Design Patterns

### 1. Strategy Pattern
- Different generation strategies for KDoc vs Tests
- Pluggable LLM backends
- Configurable embedding models

### 2. Builder Pattern
- Prompt construction with PromptBuilder
- Flexible parameter configuration
- Context-aware generation

### 3. Factory Pattern
- Component initialization
- Model loading and configuration
- Index creation

## Performance Considerations

### Optimization Strategies

1. **Embedding Caching**
   - Pre-computed embeddings stored in FAISS
   - Incremental index updates
   - Memory-efficient vector storage

2. **Batch Processing**
   - Multiple files processed together
   - Batched embedding generation
   - Parallel file operations

3. **Model Efficiency**
   - Local Ollama deployment
   - Optimized inference parameters
   - Memory management

### Scalability

- **Horizontal**: Multiple worker processes
- **Vertical**: GPU acceleration for embeddings
- **Storage**: Persistent FAISS indices
- **Caching**: Model and embedding caches

## Security Considerations

### Data Protection
- Local processing (no cloud dependencies)
- Secure file handling
- Input validation and sanitization

### Model Security
- Local LLM deployment
- No external API dependencies
- Controlled model access

## Configuration

### Environment Variables
```bash
# Ollama Configuration
OLLAMA_API_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=codellama:instruct

# Embedding Configuration
EMBEDDING_MODEL=microsoft/codebert-base
EMBEDDING_CACHE_DIR=./cache/embeddings

# Processing Configuration
BATCH_SIZE=10
MAX_CONTEXT_LENGTH=4096
```

### File Structure
```
DevTools/
├── src/
│   ├── input-src/          # Input Kotlin files
│   ├── output-test/        # Generated test files
│   ├── testcase-datastore/# Existing tests for reference
│   └── *.py               # Core modules
├── cache/
│   ├── embeddings/        # Cached embeddings
│   └── models/           # Model cache
└── config/
    └── settings.json     # Configuration
```

## Extension Points

### Custom Models
- Plugin architecture for different LLMs
- Custom embedding models
- Specialized prompt templates

### Custom Generators
- Domain-specific generators
- Custom output formats
- Integration with build systems

### Custom Indexers
- Different similarity metrics
- Custom preprocessing
- Domain-specific embeddings

## Monitoring & Logging

### Metrics
- Generation success rates
- Processing times
- Model performance
- Index efficiency

### Logging
- Structured logging with levels
- Performance metrics
- Error tracking
- Generation quality metrics

## Future Enhancements

### Planned Features
1. **Multi-language Support**: Java, Scala support
2. **IDE Integration**: IntelliJ plugin
3. **Continuous Learning**: Model fine-tuning
4. **Quality Metrics**: Automated quality assessment
5. **Batch Processing**: Large codebase handling
6. **API Server**: REST API for integrations

### Research Areas
- Code-specific embedding models
- Multi-modal code understanding
- Automated test quality assessment
- Integration with CI/CD pipelines
