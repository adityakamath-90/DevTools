# AI-Powered Kotlin Test Generation System v2.0 - Visual Diagrams

This document contains Mermaid diagrams representing the new modular system architecture, component interactions, and data flow for the AI-powered Kotlin test generation system v2.0.

## New Modular System Architecture Overview

```mermaid
graph TB
    subgraph "CLI Layer"
        MAIN[main.py<br/>Unified CLI Interface]
        LEGACY1[TestCaseGenerator.py<br/>Legacy Wrapper]
        LEGACY2[KdocGenerator.py<br/>Legacy Wrapper]
    end

    subgraph "Configuration Layer"
        CONFIG[Configuration Management<br/>src/config/settings.py]
        GENCONFIG[GenerationConfig<br/>Environment Overrides]
        LLMCONFIG[LLMConfig<br/>API Settings]
        EMBEDCONFIG[EmbeddingConfig<br/>Model Settings]
    end

    subgraph "Core Business Logic"
        TESTGEN[Test Generator<br/>src/core/test_generator.py]
        PARSER[Code Parser<br/>src/core/code_parser.py]
        PROMPTBUILDER[Prompt Builder<br/>src/core/prompt_builder.py]
    end

    subgraph "Service Layer"
        LLMSERVICE[LLM Service<br/>src/services/llm_service.py]
        EMBEDSERVICE[Embedding Service<br/>src/services/embedding_service.py]
        SIMPLESERVICE[Simple Embedding Service<br/>src/services/embedding_service.py]
        KDOCSERVICE[KDoc Service<br/>src/services/kdoc_service.py]
    end

    subgraph "Interface & Model Layer"
        INTERFACES[Base Interfaces<br/>src/interfaces/base_interfaces.py]
        MODELS[Data Models<br/>src/models/data_models.py]
        RESULTS[Result Objects<br/>TestGenerationResult, KDocResult]
    end

    subgraph "Utility Layer"
        LOGGING[Structured Logging<br/>src/utils/logging.py]
        HELPERS[Helper Functions<br/>Error Handling, File I/O]
    end

    subgraph "External Services"
        OLLAMA[Ollama Server<br/>CodeLlama Model<br/>Port 11434]
        CODEBERT[Microsoft CodeBERT<br/>Hugging Face Model]
        FAISS[FAISS Vector Index<br/>Similarity Search]
    end

    subgraph "Input/Output"
        INPUT[Kotlin Source Files<br/>src/input-src/]
        OUTPUT[Generated Tests<br/>output-test/]
        DATASTORE[Test Reference DB<br/>src/testcase--datastore/]
    end

    %% CLI to Configuration
    MAIN --> CONFIG
    LEGACY1 --> CONFIG
    LEGACY2 --> CONFIG

    %% Configuration to Services
    CONFIG --> LLMSERVICE
    CONFIG --> EMBEDSERVICE
    CONFIG --> KDOCSERVICE

    %% CLI to Core
    MAIN --> TESTGEN
    MAIN --> KDOCSERVICE
    LEGACY1 --> TESTGEN
    LEGACY2 --> KDOCSERVICE

    %% Core to Services
    TESTGEN --> LLMSERVICE
    TESTGEN --> EMBEDSERVICE
    TESTGEN --> PARSER
    TESTGEN --> PROMPTBUILDER
    KDOCSERVICE --> LLMSERVICE
    KDOCSERVICE --> PROMPTBUILDER

    %% Services to External
    LLMSERVICE --> OLLAMA
    EMBEDSERVICE -.-> CODEBERT
    EMBEDSERVICE -.-> FAISS
    SIMPLESERVICE --> DATASTORE

    %% Core to Interfaces/Models
    TESTGEN --> INTERFACES
    TESTGEN --> MODELS
    KDOCSERVICE --> INTERFACES
    KDOCSERVICE --> MODELS

    %% All layers to Utilities
    TESTGEN --> LOGGING
    LLMSERVICE --> LOGGING
    EMBEDSERVICE --> LOGGING
    KDOCSERVICE --> LOGGING

    %% Input/Output connections
    INPUT --> PARSER
    TESTGEN --> OUTPUT
    EMBEDSERVICE --> DATASTORE
    SIMPLESERVICE --> DATASTORE

    %% Styling
    classDef cliStyle fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
    classDef configStyle fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#fff
    classDef coreStyle fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff
    classDef serviceStyle fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#fff
    classDef interfaceStyle fill:#1abc9c,stroke:#16a085,stroke-width:2px,color:#fff
    classDef utilStyle fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px,color:#fff
    classDef externalStyle fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff
    classDef ioStyle fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff

    class MAIN,LEGACY1,LEGACY2 cliStyle
    class CONFIG,GENCONFIG,LLMCONFIG,EMBEDCONFIG configStyle
    class TESTGEN,PARSER,PROMPTBUILDER coreStyle
    class LLMSERVICE,EMBEDSERVICE,SIMPLESERVICE,KDOCSERVICE serviceStyle
    class INTERFACES,MODELS,RESULTS interfaceStyle
    class LOGGING,HELPERS utilStyle
    class OLLAMA,CODEBERT,FAISS externalStyle
    class INPUT,OUTPUT,DATASTORE ioStyle
```

## Modular Component Class Diagram

```mermaid
classDiagram
    %% Configuration Layer
    class GenerationConfig {
        +source_dir: str
        +test_dir: str
        +log_level: str
        +from_env() GenerationConfig
        +override_from_env() GenerationConfig
    }

    class LLMConfig {
        +api_url: str
        +model_name: str
        +timeout: int
        +max_retries: int
        +temperature: float
    }

    %% Interface Layer
    class BaseEmbeddingIndexer {
        <<abstract>>
        +index_files(patterns: List[str]) None
        +find_similar_content(query: str, top_k: int) List[str]
        +health_check() bool
    }

    class BaseLLMClient {
        <<abstract>>
        +generate_code(prompt: str) str
        +health_check() bool
    }

    %% Service Layer
    class LLMService {
        -config: LLMConfig
        -logger: Logger
        +__init__(config: LLMConfig)
        +generate_code(prompt: str) str
        +health_check() bool
        +get_model_info() dict
    }

    class EmbeddingIndexerService {
        -config: EmbeddingConfig
        -tokenizer: AutoTokenizer
        -model: AutoModel
        -index: faiss.IndexFlatL2
        +__init__(config: EmbeddingConfig)
        +index_files(patterns: List[str]) None
        +find_similar_content(query: str, top_k: int) List[str]
        +health_check() bool
    }

    class SimpleEmbeddingIndexerService {
        -config: EmbeddingConfig
        -test_cases: List[str]
        +__init__(config: EmbeddingConfig)
        +index_files(patterns: List[str]) None
        +find_similar_content(query: str, top_k: int) List[str]
        +health_check() bool
    }

    class KDocService {
        -llm_service: LLMService
        -config: GenerationConfig
        -logger: Logger
        +__init__(llm_service: LLMService, config: GenerationConfig)
        +generate_kdoc(kotlin_code: str) str
        +process_file(file_path: str) KDocResult
        +process_directory(directory_path: str) List[KDocResult]
    }

    %% Core Layer
    class KotlinTestGenerator {
        -config: GenerationConfig
        -llm_service: LLMService
        -embedding_service: BaseEmbeddingIndexer
        -logger: Logger
        +__init__(config, llm_service, embedding_service)
        +extract_class_name(code: str) Optional[str]
        +clean_generated_code(code: str) str
        +process_file(filepath: str) TestGenerationResult
        +generate_tests_for_all() List[TestGenerationResult]
    }

    class CodeParser {
        -config: GenerationConfig
        +__init__(config: GenerationConfig)
        +parse_kotlin_file(file_path: str) ParsedKotlinFile
        +extract_classes(code: str) List[KotlinClass]
        +remove_comments(code: str) str
    }

    class PromptBuilder {
        -config: GenerationConfig
        +__init__(config: GenerationConfig)
        +build_test_prompt(class_code: str, similar_tests: List[str]) str
        +build_kdoc_prompt(kotlin_code: str) str
    }

    %% Data Models
    class TestGenerationResult {
        +success: bool
        +input_file: str
        +output_file: str
        +class_name: Optional[str]
        +generated_code: Optional[str]
        +error_message: Optional[str]
        +processing_time: Optional[float]
        +similar_tests_found: int
    }

    class KDocResult {
        +success: bool
        +file_path: str
        +classes_processed: int
        +functions_processed: int
        +error_message: Optional[str]
        +generated_kdoc: Optional[str]
    }

    %% Main Application
    class GenAIApplication {
        -config: GenerationConfig
        -llm_service: LLMService
        -embedding_service: BaseEmbeddingIndexer
        -kdoc_service: KDocService
        -test_generator: KotlinTestGenerator
        +__init__(config: Optional[GenerationConfig])
        +health_check() bool
        +generate_tests(source_dir: str) List[TestGenerationResult]
        +generate_kdoc(source_dir: str) List[KDocResult]
        +generate_all(source_dir: str) dict
    }

    %% Relationships
    BaseLLMClient <|-- LLMService
    BaseEmbeddingIndexer <|-- EmbeddingIndexerService
    BaseEmbeddingIndexer <|-- SimpleEmbeddingIndexerService
    
    GenAIApplication --> GenerationConfig
    GenAIApplication --> LLMService
    GenAIApplication --> BaseEmbeddingIndexer
    GenAIApplication --> KDocService
    GenAIApplication --> KotlinTestGenerator
    
    KotlinTestGenerator --> LLMService
    KotlinTestGenerator --> BaseEmbeddingIndexer
    KotlinTestGenerator --> TestGenerationResult
    
    KDocService --> LLMService
    KDocService --> KDocResult
    
    LLMService --> LLMConfig
    EmbeddingIndexerService --> EmbeddingConfig
    SimpleEmbeddingIndexerService --> EmbeddingConfig
```
        +__init__(test_dir, embedding_model_name)
        +_load_and_index()
        +_encode(texts: List~str~) torch.Tensor
        +retrieve_similar(code: str, top_k: int) List~str~
    }

    class SimpleEmbeddingIndexer {
        -test_dir: str
        -test_cases: List~str~
        +__init__(test_dir)
        +_load_test_cases()
        +retrieve_similar(code: str, top_k: int) List~str~
    }

    class PromptBuilder {
        +build_generation_prompt(class_name, class_code, similar_tests) str
        +generate_accurate_prompt(class_code, generated_test) str
    }

    class KotlinTestGenerator {
        -source_dir: str
        -test_dir: str
        -llm_client: LLMClient
        -indexer: EmbeddingIndexer|SimpleEmbeddingIndexer
        +__init__(source_dir, test_dir, llm_client, indexer)
        +extract_class_name(code: str) Optional~str~
        +clean_generated_code(generated_code: str) str
        +process_file(filepath: str)
        +generate_tests_for_all()
    }

    class KdocGenerator {
        <<module>>
        +generate_kdoc_for_file(file_content: str) str
        +create_backup(filepath: str)
        +update_kdocs_in_file(filepath: str)
        +update_kdocs_in_directory(directory: str)
    }

    %% Relationships
    KotlinTestGenerator --> LLMClient : uses
    KotlinTestGenerator --> EmbeddingIndexer : uses (primary)
    KotlinTestGenerator --> SimpleEmbeddingIndexer : uses (fallback)
    KotlinTestGenerator --> PromptBuilder : uses
    EmbeddingIndexer --> "AutoTokenizer" : uses
    EmbeddingIndexer --> "AutoModel" : uses
    EmbeddingIndexer --> "faiss.IndexFlatL2" : uses
    KdocGenerator --> LLMClient : uses
    KdocGenerator --> PromptBuilder : uses
```

# Sequence Diagram - Test Generation Flow

```mermaid
sequenceDiagram
    participant User
    participant TestGen as KotlinTestGenerator
    participant Embed as EmbeddingIndexer
    participant Simple as SimpleEmbeddingIndexer
    participant Prompt as PromptBuilder
    participant LLM as LLMClient
    participant Ollama

    User->>TestGen: Run TestCaseGenerator.py
    TestGen->>TestGen: Initialize components

    alt Advanced EmbeddingIndexer available
        TestGen->>Embed: __init__(test_dir)
        Embed->>Embed: Load Microsoft CodeBERT model
        Embed->>Embed: Index existing test cases with FAISS
        Note over Embed: Creates embeddings for similarity search
    else Fallback to SimpleEmbeddingIndexer
        TestGen->>Simple: __init__(test_dir)
        Simple->>Simple: Load test cases as plain text
        Note over Simple: Simple text-based matching
    end

    TestGen->>TestGen: generate_tests_for_all()
    TestGen->>TestGen: Scan src/input-src/ directory
    
    loop For each Kotlin file
        TestGen->>TestGen: extract_class_name(file_content)
        
        alt Class found
            alt Using Advanced Indexer
                TestGen->>Embed: retrieve_similar(file_content)
                Embed->>Embed: Encode input with CodeBERT
                Embed->>Embed: Search FAISS index for similar patterns
                Embed-->>TestGen: Top-K similar test cases
            else Using Simple Indexer
                TestGen->>Simple: retrieve_similar(file_content)
                Simple->>Simple: Return first K test cases
                Simple-->>TestGen: Available test cases
            end
            
            TestGen->>Prompt: build_generation_prompt(class_name, file_content, similar_tests)
            Prompt-->>TestGen: Structured generation prompt
            
            TestGen->>LLM: generate(generation_prompt)
            LLM->>Ollama: POST /api/generate (CodeLlama)
            Ollama-->>LLM: Generated test code
            LLM-->>TestGen: Raw generated test
            
            alt Test generation successful
                TestGen->>TestGen: clean_generated_code()
                Note over TestGen: Remove markdown formatting
                
                TestGen->>Prompt: generate_accurate_prompt(file_content, generated_test)
                Prompt-->>TestGen: Accuracy validation prompt
                
                TestGen->>LLM: generate(accuracy_prompt)
                LLM->>Ollama: POST /api/generate
                Ollama-->>LLM: Feedback/improvements
                LLM-->>TestGen: Validation feedback
                
                TestGen->>TestGen: Write test file to output-test/
                Note over TestGen: Save as {ClassName}Test.kt
            else Generation failed
                TestGen->>TestGen: Log error and continue
            end
        else No class found
            TestGen->>TestGen: Skip file with warning
        end
    end
    
    TestGen-->>User: Generation complete with summary
```

# Updated Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV[Developer Machine<br/>macOS/Linux/Windows]
        IDE[IDE/Editor<br/>VS Code/PyCharm/Terminal]
        PYTHON[Python 3.9+<br/>Virtual Environment]
    end

    subgraph "Local AI Infrastructure"
        OLLAMA_SERVER[Ollama Server<br/>localhost:11434]
        CODELLAMA[CodeLlama Model<br/>codellama:instruct]
        MODEL_CACHE[Model Cache<br/>~/.ollama/models]
    end

    subgraph "Python ML Stack"
        TORCH[PyTorch<br/>Deep Learning Framework]
        TRANSFORMERS[Transformers<br/>Hugging Face Library]
        CODEBERT_LOCAL[CodeBERT Model<br/>microsoft/codebert-base]
        FAISS_LIB[FAISS Library<br/>Vector Similarity Search]
    end

    subgraph "Application Environment"
        VENV[Virtual Environment<br/>.venv/]
        REQUIREMENTS[Dependencies<br/>requirements.txt]
        DEVTOOLS[DevTools Application<br/>Python Scripts]
    end

    subgraph "File System Structure"
        INPUT_DIR[Input Directory<br/>src/input-src/]
        OUTPUT_DIR[Output Directory<br/>output-test/]
        REFERENCE_DIR[Reference Tests<br/>src/testcase--datastore/]
        DOCS_DIR[Documentation<br/>docs/]
    end

    subgraph "Runtime Data"
        EMBEDDINGS_CACHE[Embeddings Cache<br/>In-Memory Tensors]
        FAISS_INDEX[FAISS Index<br/>Vector Database]
        BACKUPS[File Backups<br/>*.backup files]
    end

    %% Development Flow
    DEV --> IDE
    IDE --> PYTHON
    PYTHON --> VENV
    VENV --> REQUIREMENTS
    REQUIREMENTS --> DEVTOOLS

    %% AI Infrastructure
    DEVTOOLS --> OLLAMA_SERVER
    OLLAMA_SERVER --> CODELLAMA
    CODELLAMA --> MODEL_CACHE

    %% ML Stack Integration
    DEVTOOLS --> TORCH
    DEVTOOLS --> TRANSFORMERS
    TRANSFORMERS --> CODEBERT_LOCAL
    DEVTOOLS --> FAISS_LIB

    %% File System Access
    DEVTOOLS --> INPUT_DIR
    DEVTOOLS --> OUTPUT_DIR
    DEVTOOLS --> REFERENCE_DIR
    DEVTOOLS --> DOCS_DIR

    %% Runtime Data Management
    DEVTOOLS --> EMBEDDINGS_CACHE
    DEVTOOLS --> FAISS_INDEX
    DEVTOOLS --> BACKUPS

    %% Styling
    classDef devStyle fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
    classDef aiStyle fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff
    classDef mlStyle fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#fff
    classDef appStyle fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#fff
    classDef dataStyle fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff
    classDef runtimeStyle fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px,color:#fff

    class DEV,IDE,PYTHON devStyle
    class OLLAMA_SERVER,CODELLAMA,MODEL_CACHE aiStyle
    class TORCH,TRANSFORMERS,CODEBERT_LOCAL,FAISS_LIB mlStyle
    class VENV,REQUIREMENTS,DEVTOOLS appStyle
    class INPUT_DIR,OUTPUT_DIR,REFERENCE_DIR,DOCS_DIR dataStyle
    class EMBEDDINGS_CACHE,FAISS_INDEX,BACKUPS runtimeStyle
```

## Data Flow Architecture

```mermaid
graph LR
    subgraph "Input Processing"
        KT_FILES[Kotlin Files<br/>*.kt]
        CLASS_EXTRACT[Class Extraction<br/>Regex Pattern Matching]
        CONTENT_CLEAN[Content Cleaning<br/>Comment Removal]
    end

    subgraph "Similarity Analysis"
        ENCODE[Text Encoding<br/>CodeBERT Embeddings]
        VECTOR_SEARCH[Vector Search<br/>FAISS Index]
        CONTEXT_MATCH[Context Matching<br/>Top-K Similar Tests]
    end

    subgraph "AI Generation"
        PROMPT_BUILD[Prompt Construction<br/>Context + Instructions]
        LLM_CALL[LLM Generation<br/>Ollama + CodeLlama]
        CODE_CLEAN[Code Cleaning<br/>Markdown Removal]
    end

    subgraph "Quality Assurance"
        VALIDATION[Test Validation<br/>Accuracy Checking]
        FEEDBACK[Feedback Loop<br/>Improvement Suggestions]
        FINAL_CLEAN[Final Cleaning<br/>Format Standardization]
    end

    subgraph "Output Generation"
        FILE_WRITE[File Writing<br/>Test File Creation]
        BACKUP_CREATE[Backup Creation<br/>Safety Measures]
        LOGGING[Progress Logging<br/>Status Updates]
    end

    %% Data Flow
    KT_FILES --> CLASS_EXTRACT
    CLASS_EXTRACT --> CONTENT_CLEAN
    CONTENT_CLEAN --> ENCODE
    ENCODE --> VECTOR_SEARCH
    VECTOR_SEARCH --> CONTEXT_MATCH
    CONTEXT_MATCH --> PROMPT_BUILD
    PROMPT_BUILD --> LLM_CALL
    LLM_CALL --> CODE_CLEAN
    CODE_CLEAN --> VALIDATION
    VALIDATION --> FEEDBACK
    FEEDBACK --> FINAL_CLEAN
    FINAL_CLEAN --> FILE_WRITE
    FILE_WRITE --> BACKUP_CREATE
    BACKUP_CREATE --> LOGGING

    %% Styling
    classDef inputStyle fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
    classDef processStyle fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#fff
    classDef aiStyle fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff
    classDef qaStyle fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#fff
    classDef outputStyle fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff

    class KT_FILES,CLASS_EXTRACT,CONTENT_CLEAN inputStyle
    class ENCODE,VECTOR_SEARCH,CONTEXT_MATCH processStyle
    class PROMPT_BUILD,LLM_CALL,CODE_CLEAN aiStyle
    class VALIDATION,FEEDBACK,FINAL_CLEAN qaStyle
    class FILE_WRITE,BACKUP_CREATE,LOGGING outputStyle
```

---

*Last Updated: July 3, 2025*  
*These diagrams reflect the current implementation with Microsoft CodeBERT embedding support and fallback mechanisms.*

## Modular Data Flow Sequence Diagram

```mermaid
sequenceDiagram
    participant CLI as CLI Interface
    participant App as GenAIApplication
    participant Config as Configuration
    participant TestGen as KotlinTestGenerator
    participant Parser as CodeParser
    participant EmbedSvc as EmbeddingService
    participant LLMSvc as LLMService
    participant PromptBld as PromptBuilder
    participant Logger as Logging System
    participant FileIO as File System

    Note over CLI, FileIO: Test Generation Flow (New Architecture)

    CLI->>App: generate_tests(source_dir)
    App->>Config: Load configuration with env overrides
    Config-->>App: Return GenerationConfig
    
    App->>Logger: Log start of generation process
    App->>TestGen: Initialize with services
    
    TestGen->>Parser: parse_kotlin_file(file_path)
    Parser->>FileIO: Read Kotlin source file
    FileIO-->>Parser: File content
    Parser->>Parser: remove_comments(code)
    Parser->>Parser: extract_classes(code)
    Parser-->>TestGen: ParsedKotlinFile
    
    TestGen->>EmbedSvc: find_similar_content(class_name)
    
    alt Advanced Embedding Available
        EmbedSvc->>EmbedSvc: Use CodeBERT + FAISS
        EmbedSvc-->>TestGen: List of similar test patterns
    else Fallback to Simple
        EmbedSvc->>EmbedSvc: Use simple text matching
        EmbedSvc-->>TestGen: List of similar test patterns
    end
    
    TestGen->>PromptBld: build_test_prompt(class_code, similar_tests)
    PromptBld-->>TestGen: Structured prompt with context
    
    TestGen->>LLMSvc: generate_code(prompt)
    
    alt LLM Service Available
        LLMSvc->>LLMSvc: Call Ollama API
        LLMSvc-->>TestGen: Generated test code
    else LLM Service Unavailable
        LLMSvc->>Logger: Log error and fallback
        LLMSvc-->>TestGen: Error response
    end
    
    TestGen->>TestGen: clean_generated_code(raw_code)
    TestGen->>FileIO: Write test file to output directory
    TestGen->>Logger: Log successful generation
    
    TestGen-->>App: TestGenerationResult with metadata
    App->>Logger: Log completion
    App-->>CLI: List[TestGenerationResult]
```

## Configuration and Health Check Flow

```mermaid
sequenceDiagram
    participant CLI as CLI Interface
    participant App as GenAIApplication
    participant Config as Configuration
    participant LLMSvc as LLMService
    participant EmbedSvc as EmbeddingService
    participant KDocSvc as KDocService
    participant Logger as Logging System

    Note over CLI, Logger: Health Check and Initialization Flow

    CLI->>App: health_check()
    App->>Config: Load environment configuration
    Config->>Config: Validate and merge env variables
    Config-->>App: Validated configuration
    
    App->>Logger: Initialize structured logging
    
    par Service Health Checks
        App->>LLMSvc: health_check()
        LLMSvc->>LLMSvc: Test Ollama connection
        LLMSvc-->>App: Service status
    and
        App->>EmbedSvc: health_check()
        EmbedSvc->>EmbedSvc: Test model loading
        EmbedSvc-->>App: Service status
    and
        App->>KDocSvc: health_check()
        KDocSvc->>KDocSvc: Validate dependencies
        KDocSvc-->>App: Service status
    end
    
    App->>Logger: Log health check results
    App-->>CLI: Overall system health status
```

## Error Handling and Fallback Flow

```mermaid
flowchart TD
    A[Request Received] --> B{Configuration Valid?}
    B -->|No| C[Log Error & Return Config Error]
    B -->|Yes| D[Initialize Services]
    
    D --> E{LLM Service Available?}
    E -->|No| F[Log Warning & Continue with Fallback]
    E -->|Yes| G[Proceed with LLM Generation]
    
    D --> H{Advanced Embedding Available?}
    H -->|No| I[Use Simple Embedding Service]
    H -->|Yes| J[Use CodeBERT Embedding Service]
    
    I --> K[Text-based Similarity Matching]
    J --> L{CodeBERT Loading Successful?}
    L -->|No| M[Fallback to Simple Service]
    L -->|Yes| N[Semantic Similarity with FAISS]
    
    M --> K
    N --> O[Context-aware Prompt Building]
    K --> O
    
    F --> P[Generate Simple Template]
    G --> Q{LLM Response Valid?}
    Q -->|No| R[Log Error & Use Template]
    Q -->|Yes| S[Process Generated Code]
    
    O --> G
    P --> T[Return Result with Warnings]
    R --> T
    S --> U[Clean and Validate Code]
    U --> V[Write Output Files]
    V --> W[Return Success Result]
    
    style A fill:#3498db
    style C fill:#e74c3c
    style F fill:#f39c12
    style T fill:#f39c12
    style W fill:#2ecc71
```
