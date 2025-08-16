# 📚 Project Documentation

Welcome to the DevTools documentation hub. This directory contains comprehensive documentation for the DevTools project.

## 📋 Table of Contents

### Core Documentation
- [API Reference](API.md) — Module and function overview
- [Architecture Overview](ARCHITECTURE.md) — System design and key components
- [Diagrams](DIAGRAMS.md) — Data flow and component diagrams

## 🔍 Getting Started

1. Read the [Architecture Overview](ARCHITECTURE.md)
2. Ensure you have Python 3.10+ and (optionally) a local LLM via Ollama
3. Install deps: `pip install -r requirements.txt`
4. Run the Web UI: `streamlit run src/ui/webui.py`
5. Or run the CLI pipeline: `python main.py`

### LLM Setup (Local)
- Install Ollama and pull a model, e.g.: `ollama pull codellama:7b`
- Configure in code via `LangChainOllamaProvider` (see `src/providers/langchain_provider.py`) or use the lightweight agent in `src/services/llm_agent.py`.

#### Lightweight Agent Quickstart
```python
from src.services.llm_agent import CodeLlamaAgent, TestGeneratorAgent
from src.core.prompt_builder import PromptBuilder

class_code = """
class Calculator {
    fun add(a: Int, b: Int) = a + b
}
"""

agent = CodeLlamaAgent(api_url="http://127.0.0.1:11434", model="codellama:instruct")
tg = TestGeneratorAgent(agent)
pb = PromptBuilder()

prompt = pb.build_test_prompt(class_code, similar_tests=[])
tests = tg.generate_tests("Calculator", class_code, similar_tests=[])
print(tests)
```

## 🤝 Contributing to Documentation

We welcome contributions to improve our documentation! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details on how to contribute.

## 📝 Documentation Standards

- Use Markdown (.md) for all documentation
- Follow the [Google Developer Documentation Style Guide](https://developers.google.com/style)
- Include code examples where appropriate
- Keep diagrams in the `diagrams/` directory
- Update the relevant documentation when making code changes

## 🔗 Related Resources

- [Project Wiki](https://github.com/yourusername/DevTools/wiki) — Community-maintained docs
- Changelog — See repository releases and commit history

---

## 🗂️ Source Code Overview (`src/`)

High-level map of the application source tree and responsibilities:

- `src/agents/test_pipeline.py`
  Orchestrates the test generation pipeline across stages.

- `src/config/`
  Runtime configuration. Includes `settings.py` and `langchain_config.py`.

- `src/core/`
  Core logic for parsing and prompt construction.
  - `code_parser.py` — Kotlin source parsing utilities
  - `prompt_builder.py` — Prompt assembly for LLMs
  - `test_generator.py` — Test generation and optional validation hooks

- `src/interfaces/`
  Abstract base interfaces (e.g., `LLMProvider`, `SimilarityIndexer`).

- `src/models/data_models.py`
  Typed models for requests, responses, and metrics.

- `src/providers/`
  Concrete LLM providers.
  - `default_provider.py` — Default HTTP/OpenAI-style provider
  - `langchain_provider.py` — LangChain + Ollama provider (`LangChainOllamaProvider`)

- `src/services/`
  Supporting services.
  - `llm_service.py` — High-level LLM service abstraction and provider wiring
  - `embedding_service.py` — Embedding-based similarity (CodeBERT/FAISS and simple fallback)
  - `llm_agent.py` — Lightweight local CodeLlama agent and `TestGeneratorAgent` facade
  - `kdoc_service.py` — KDoc extraction helpers (optional)

- `src/ui/webui.py`
  Streamlit Web UI for interactive test generation.

- `src/utils/`
  Utilities (logging, helpers). Includes `utils/logging.py`.

Note: Some auxiliary modules may be optional depending on your workflow (e.g., `kdoc_service.py`).
