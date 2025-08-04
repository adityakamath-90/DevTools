# System Architecture & Data Flow Diagrams

This document provides visual representations of the Kotlin Test Generation System's architecture and data flows using Mermaid diagrams.

## System Architecture Overview

```mermaid
graph TD
    %% Main Components
    CLI[DevTools CLI] --> Core[Core Services]
    Core --> AI[AI Services]
    Core --> Validation[Validation System]
    
    %% Core Services
    Core --> TestGen[Test Generator]
    Core --> PromptBuilder[Prompt Builder]
    Core --> Parser[Code Parser]
    Core --> FileMgr[File Manager]
    
    %% AI Services
    AI --> LLM[LLM Service]
    AI --> Embedding[Embedding Service]
    AI --> KDoc[KDoc Service]
    
    %% External Dependencies
    LLM --> Ollama[Ollama/CodeLlama]
    Embedding --> FAISS[FAISS Index]
    Embedding --> CodeBERT[CodeBERT]
    
    %% Data Flow
    CLI -->|source_dir, config| Core
    TestGen -->|parsed code| PromptBuilder
    PromptBuilder -->|prompt| LLM
    LLM -->|generated tests| FileMgr
    Parser -->|code analysis| Embedding
    Embedding -->|similar tests| PromptBuilder
    FileMgr -->|test files| Validation
    
    classDef component fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef service fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef external fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
    
    class CLI,Core,TestGen,PromptBuilder,Parser,FileMgr component
    class AI,LLM,Embedding,KDoc service
    class Ollama,FAISS,CodeBERT external
```

## Data Flow - Test Generation Process

```mermaid
flowchart TD
    A[Input Kotlin Files] --> B[Code Parser]
    B --> C[Extract Classes & Methods]
    C --> D[Find Similar Tests]
    D --> E[Build Generation Prompt]
    E --> F[Generate with LLM]
    F --> G[Post-Process Output]
    G --> H[Save Test Files]
    H --> I[Run Validations]
    
    subgraph AI_Components[AI Components]
        D -->|get embeddings| D1[FAISS Index]
        D1 -->|return similar| D
        F -->|API Request| F1[Ollama/CodeLlama]
        F1 -->|Generated Code| F
    end
    
    style A fill:#e8f5e9,stroke:#2e7d32
    style I fill:#ffebee,stroke:#c62828
    style AI_Components fill:#e8eaf6,stroke:#3949ab
```

## Sequence Diagram - Test Generation

```mermaid
sequenceDiagram
    participant User
    participant CLI as Command Line
    participant Core as Test Generator
    participant AI as AI Services
    participant Val as Validator
    
    User->>+CLI: python main.py generate-tests
    
    CLI->>+Core: Initialize services
    Core-->>-CLI: Ready
    
    loop For each Kotlin file
        Core->>+Core: Parse source code
        Core->>+AI: Get embeddings
        AI-->>-Core: Similar tests
        
        Core->>+AI: Create prompt
        AI-->>-Core: Generated prompt
        
        Core->>+AI: Generate tests
        AI->>+LLM: API Request
        LLM-->>-AI: Test code
        AI-->>-Core: Formatted tests
        
        Core->>+Val: Validate tests
        Val-->>-Core: Results
        
        Core->>Core: Save tests
        Core->>CLI: Update progress
    end
    
    CLI->>+Val: Final validation
    Val-->>-CLI: Summary
    CLI-->>-User: Complete
```

## Component Interaction

```mermaid
graph LR
    subgraph Core[Core Components]
        TG[TestGenerator]
        PB[PromptBuilder]
        LS[LLM Service]
        ES[Embedding Service]
        
        TG -->|uses| PB
        TG -->|manages| LS
        TG -->|queries| ES
    end
    
    subgraph Support[Support Services]
        CM[Config Manager]
        LOG[Logger]
        FM[File Manager]
        
        CM -->|configures| TG
        LOG -->|logs| TG
        FM -->|handles I/O| TG
    end
    
    subgraph Ext[External]
        OL[Ollama]
        FS[FAISS]
        CB[CodeBERT]
        
        OL -->|powers| LS
        FS -->|indexes| ES
        CB -->|embeds| ES
    end
    
    style Core fill:#e1f5fe,stroke:#0288d1
    style Support fill:#e8f5e9,stroke:#2e7d32
    style Ext fill:#f3e5f5,stroke:#8e24aa
        TestGen->>LLM: generate(prompt)
        LLM-->>TestGen: test code
        TestGen->>TestGen: clean_generated_code()
        TestGen->>File: save test file
    end
    TestGen-->>CLI: results
    CLI-->>User: summary
```

## Component Relationships

```mermaid
classDiagram
    class main_py {
        +GenAIApplication
    }
    class TestGenerator {
        +generate_tests_for_directory()
        +generate_tests()
    }
    class PromptBuilder {
        +build_generation_prompt()
    }
    class LLMService {
        +generate()
    }
    class EmbeddingService {
        +find_similar()
    }
    class KDocService {
        +generate_kdoc()
    }
    main_py --> TestGenerator
    TestGenerator --> PromptBuilder
    TestGenerator --> LLMService
    TestGenerator --> EmbeddingService
    TestGenerator --> KDocService
```

---

*Last updated: July 2025. For more, see API and ARCHITECTURE docs.*
