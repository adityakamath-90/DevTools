# AI-Powered Kotlin Test Generation System - Visual Diagrams

This document contains Mermaid diagrams representing the system architecture, component interactions, and data flow for the AI-powered Kotlin test generation system.

## System Architecture Overview

```mermaid
graph TB
    subgraph "Input Layer"
        KT[Kotlin Source Files<br/>src/input-src/]
        CONFIG[Configuration<br/>Environment Variables]
    end

    subgraph "Application Layer"
        KDOCGEN[KDoc Generator<br/>KdocGenerator.py]
        TESTGEN[Test Case Generator<br/>TestCaseGenerator.py]
    end

    subgraph "Core Services"
        LLM[LLM Client<br/>LLMClient.py]
        EMBED[Embedding Indexer<br/>EmbeddingIndexer.py]
        SIMPLE[Simple Embedding Indexer<br/>SimpleEmbeddingIndexer.py]
        PROMPT[Prompt Builder<br/>PromptBuilder.py]
    end

    subgraph "External Services"
        OLLAMA[Ollama Server<br/>CodeLlama Model<br/>Port 11434]
        CODEBERT[Microsoft CodeBERT<br/>Hugging Face Model]
        FAISS[FAISS Vector Index<br/>Similarity Search]
    end

    subgraph "Output Layer"
        KDOCOUT[Enhanced Kotlin Files<br/>with KDoc Comments]
        TESTOUT[JUnit 5 Test Files<br/>output-test/]
        DATASTORE[Test Reference Database<br/>src/testcase--datastore/]
    end

    %% Input connections
    KT --> KDOCGEN
    KT --> TESTGEN
    CONFIG --> LLM

    %% Application to Services
    KDOCGEN --> LLM
    KDOCGEN --> PROMPT
    TESTGEN --> LLM
    TESTGEN --> EMBED
    TESTGEN --> SIMPLE
    TESTGEN --> PROMPT

    %% Services to External (with fallback)
    LLM --> OLLAMA
    EMBED -.-> CODEBERT
    EMBED -.-> FAISS
    SIMPLE --> DATASTORE

    %% Cross-service communication
    EMBED --> PROMPT
    SIMPLE --> PROMPT
    PROMPT --> LLM

    %% Output connections
    KDOCGEN --> KDOCOUT
    TESTGEN --> TESTOUT
    EMBED --> DATASTORE
    SIMPLE --> DATASTORE

    %% Styling
    classDef inputStyle fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
    classDef appStyle fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff
    classDef serviceStyle fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#fff
    classDef externalStyle fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#fff
    classDef outputStyle fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff

    class KT,CONFIG inputStyle
    class KDOCGEN,TESTGEN appStyle
    class LLM,EMBED,SIMPLE,PROMPT serviceStyle
    class OLLAMA,CODEBERT,FAISS externalStyle
    class KDOCOUT,TESTOUT,DATASTORE outputStyle
```

## Component Class Diagram

```mermaid
classDiagram
    class LLMClient {
        -api_url: str
        -model_name: str
        +__init__(api_url, model_name)
        +generate(prompt: str) str
    }

    class EmbeddingIndexer {
        -test_dir: str
        -tokenizer: AutoTokenizer
        -model: AutoModel
        -test_cases: List~str~
        -embeddings: torch.Tensor
        -index: faiss.IndexFlatL2
        -dimension: int
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
