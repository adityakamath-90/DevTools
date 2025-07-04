# AI-Powered Kotlin Test Generation System v2.0 - Visual Diagrams

This document contains Mermaid diagrams representing the new modular system architecture, component interactions, and data flow for the AI-powered Kotlin test generation system v2.0.

## Diagrams

## System Overview

```mermaid
graph TD
    A[main.py CLI] --> B[Test Generator]
    B --> C[Prompt Builder]
    B --> D[LLM Service]
    B --> E[Embedding Service]
    B --> F[KDoc Service]
    B --> G[Output Files]
    C --> D
    D --> G
    E --> B
    F --> G
```

## Data Flow

1. User runs CLI command.
2. Test generator parses source and finds similar tests.
3. Prompt builder constructs LLM prompt.
4. LLM service generates test code.
5. Test code is validated and saved.
6. Output files are written to disk.

## Sequence Diagram - Test Generation

```mermaid
sequenceDiagram
    participant User
    participant CLI as main.py
    participant TestGen as TestGenerator
    participant Embed as EmbeddingService
    participant Prompt as PromptBuilder
    participant LLM as LLMService
    participant File as OutputFile

    User->>CLI: Run test command
    CLI->>TestGen: generate_tests_for_directory()
    loop For each Kotlin file
        TestGen->>Embed: find_similar()
        Embed-->>TestGen: similar tests
        TestGen->>Prompt: build_generation_prompt()
        Prompt-->>TestGen: prompt
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
