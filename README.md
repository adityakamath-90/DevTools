# 🚀 DevTools: AI-Powered Development Assistant

A comprehensive suite of developer tools that leverages AI to enhance software development workflows, with a focus on test generation, code documentation, and code quality.

## ✨ Key Features

- **🤖 AI-Powered Test Generation**: Generate JUnit 5 test cases for Kotlin code using CodeLlama
- **📝 Smart Documentation**: Automatic KDoc comment generation
- **🔍 Semantic Search**: Find relevant test patterns using CodeBERT embeddings
- **⚡ Fast & Efficient**: Optimized with FAISS for quick similarity searches
- **🧩 Modular Design**: Clean, maintainable architecture with clear separation of concerns

## 🏗️ Project Structure

```
DevTools/
├── src/                    # Source code
│   ├── core/              # Core business logic
│   ├── services/          # Service implementations
│   ├── interfaces/        # Abstract interfaces
│   ├── models/            # Data models
│   ├── config/            # Configuration
│   └── utils/             # Utility functions
├── docs/                  # Documentation
├── validation-system/     # Test validation
└── tests/                 # Test files
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Ollama with CodeLlama model
- Java Development Kit (JDK) 11+
- Gradle 7.0+

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/DevTools.git
   cd DevTools
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Ollama**:
   ```bash
   # Install Ollama if not present
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Download CodeLlama model
   ollama pull codellama:instruct
   ```

## 🛠️ Usage

### Generate Tests

```bash
# Generate tests for all Kotlin files in src/input-src/
python main.py generate-tests

# Generate tests for a specific file
python main.py generate-tests --source-file path/to/YourFile.kt
```

### Generate Documentation

```bash
# Generate KDoc comments
python main.py generate-kdoc
```

### Run Validations

```bash
# Run all validations
cd validation-system
./scripts/validate_all.sh
```

## 📚 Documentation

For detailed documentation, see the [docs](docs/) directory:
- [API Documentation](docs/API.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Validation Guide](validation-system/docs/RUNTIME_VALIDATION_GUIDE.md)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for the local LLM infrastructure
- [CodeLlama](https://ai.meta.com/llama/) for the base AI model
- [FAISS](https://faiss.ai/) for efficient similarity search
- All our amazing contributors

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** with pip installed
2. **Ollama** installed and running locally
3. **CodeLlama model** downloaded via Ollama

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DevTools
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Download CodeLlama model
   ollama pull codellama:instruct
   
   # Start Ollama server (if not running)
   ollama serve
   ```

### Usage

#### New Modular CLI (Recommended)

```bash
# Generate tests for all Kotlin files in src/input-src/
python main.py generate-tests

# Generate tests for a specific directory
python main.py generate-tests --source-dir path/to/kotlin/files

# Generate KDoc comments
python main.py generate-kdoc

# Generate both tests and KDoc
python main.py generate-all

# Health check
python main.py health-check
```

#### Legacy Compatibility

For backward compatibility, the legacy scripts still work:

```bash
# Generate tests (legacy)
python TestCaseGenerator.py

# Generate KDoc (legacy)  
python KdocGenerator.py
```

### Configuration

The system supports flexible configuration through:

1. **Environment variables**:
   ```bash
   export OLLAMA_API_URL="http://127.0.0.1:11434/api/generate"
   export MODEL_NAME="codellama:instruct"
   export EMBEDDING_MODEL="microsoft/codebert-base"
   export SOURCE_DIR="src/input-src"
   export TEST_DIR="output-test"
   ```

2. **Configuration files**: Automatically loaded from `src/config/settings.py`

3. **Command-line arguments**: Override any configuration option

### Output

- **Generated Tests**: Saved in `output-test/` directory
- **Test Files**: Named as `{ClassName}Test.kt`
- **Format**: JUnit 5 compatible Kotlin test code
- **Dependencies**: Uses MockK for mocking when needed

## 🛠️ System Features

### ✅ Current Capabilities

- **Modular Architecture**: Interface-driven design with dependency injection
- **Advanced Class Detection**: Smart regex patterns with comment filtering
- **Semantic Similarity**: Microsoft CodeBERT embeddings for context matching
- **Test Generation**: Comprehensive JUnit 5 test creation
- **Code Cleaning**: Automatic markdown removal and formatting
- **Batch Processing**: Process entire directories of Kotlin files
- **Fallback Support**: Simple indexer when advanced features unavailable
- **Documentation**: Automatic KDoc comment generation
- **Robust Configuration**: Environment-based settings with overrides
- **Structured Logging**: Comprehensive logging system with different levels
- **Error Handling**: Graceful degradation and recovery mechanisms

### 🔧 Technical Stack

- **Architecture**: Modular service-oriented design
- **AI Model**: CodeLlama (via Ollama)
- **Embeddings**: Microsoft CodeBERT
- **Vector Search**: FAISS
- **Configuration**: Dataclass-based with environment overrides
- **Logging**: Structured logging with configurable levels
- **Language**: Python 3.9+
- **Target**: Kotlin with JUnit 5

## 📁 Project Structure

```
DevTools/
├── main.py                  # New unified CLI entry point
├── TestCaseGenerator.py     # Legacy test generation (backward compatibility)
├── KdocGenerator.py         # Legacy KDoc generation (backward compatibility)
├── requirements.txt         # Python dependencies
├── src/
│   ├── config/             # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py     # Environment-based configuration
│   ├── core/               # Core business logic
│   │   ├── __init__.py
│   │   ├── code_parser.py  # Kotlin code parsing
│   │   ├── prompt_builder.py # Context-aware prompt construction
│   │   └── test_generator.py # Test generation orchestrator
│   ├── services/           # Service layer
│   │   ├── __init__.py
│   │   ├── llm_service.py  # LLM interface with error handling
│   │   ├── embedding_service.py # Semantic similarity service
│   │   └── kdoc_service.py # KDoc generation service
│   ├── interfaces/         # Abstract base classes
│   │   ├── __init__.py
│   │   └── base_interfaces.py
│   ├── models/             # Data models
│   │   ├── __init__.py
│   │   └── data_models.py  # Result objects and data structures
│   ├── utils/              # Utilities
│   │   ├── __init__.py
│   │   └── logging.py      # Structured logging
│   └── input-src/          # Input Kotlin source files
│       ├── Calculator.kt   # Sample calculator class
│       └── UserManager.kt  # Sample user management class
├── output-test/            # Generated test files (created automatically)
├── docs/                   # Comprehensive documentation
│   ├── ARCHITECTURE.md     # System architecture
│   ├── API.md             # API documentation
│   ├── DIAGRAMS.md        # Visual diagrams
│   └── MIGRATION.md       # Migration guide
└── README.md              # This file
```

## 🔄 Migration Guide

### From Legacy to Modular Architecture

If you're upgrading from the legacy version:

1. **No immediate changes required** - Legacy scripts (`TestCaseGenerator.py`, `KdocGenerator.py`) still work
2. **Recommended**: Switch to the new CLI (`python main.py`) for better features
3. **Configuration**: Move environment variables to the new format (see Configuration section)
4. **Custom integrations**: Use the new service-based API (see `docs/API.md`)

For detailed migration instructions, see [MIGRATION.md](./MIGRATION.md).

## 📁 Project Structure

```
DevTools/
├── src/
│   ├── input-src/           # Input Kotlin source files
│   │   ├── Calculator.kt    # Sample calculator class
│   │   └── UserManager.kt   # Sample user management class
│   ├── testcase--datastore/ # Existing test cases for context
│   ├── TestCaseGenerator.py # Main test generation script
│   ├── EmbeddingIndexer.py  # Semantic similarity indexing
│   ├── LLMClient.py         # Ollama/CodeLlama interface
│   ├── PromptBuilder.py     # Context-aware prompt construction
│   └── KdocGenerator.py     # KDoc comment generation
├── output-test/             # Generated test files (created automatically)
├── docs/                    # Detailed documentation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```
## � Configuration

### Environment Variables

The system supports the following environment variables:

```bash
# Ollama API Configuration
export OLLAMA_API_URL="http://127.0.0.1:11434/api/generate"
export MODEL_NAME="codellama:instruct"

# Embedding Model Configuration
export EMBEDDING_MODEL="microsoft/codebert-base"
```

### Model Options

The system automatically handles model selection:

1. **Primary**: Microsoft CodeBERT for embeddings + CodeLlama for generation
2. **Fallback**: Simple text matching when advanced models unavailable
3. **Configurable**: Change models via environment variables

## 📊 Example Workflow

### Input: Calculator.kt
```kotlin
class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }
    
    fun divide(a: Int, b: Int): Int {
        if (b == 0) throw IllegalArgumentException("Division by zero")
        return a / b
    }
}
```

### Generated Output: CalculatorTest.kt
```kotlin
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.assertThrows

class CalculatorTest {
    
    @Test
    fun testAdd() {
        val calculator = Calculator()
        assertEquals(5, calculator.add(2, 3))
        assertEquals(0, calculator.add(-1, 1))
        assertEquals(-5, calculator.add(-2, -3))
    }
    
    @Test
    fun testDivide() {
        val calculator = Calculator()
        assertEquals(2, calculator.divide(6, 3))
        assertEquals(0, calculator.divide(0, 5))
    }
    
    @Test
    fun testDivideByZero() {
        val calculator = Calculator()
        assertThrows<IllegalArgumentException> {
            calculator.divide(10, 0)
        }
    }
}
```

## 🧪 Testing the System

### Test with Sample Files

1. **Health check** (verify all services are working):
   ```bash
   python main.py health-check
   ```

2. **Run test generation** (new CLI):
   ```bash
   python main.py generate-tests
   ```

3. **Check generated tests**:
   ```bash
   ls -la output-test/
   cat output-test/CalculatorTest.kt
   ```

4. **Generate documentation**:
   ```bash
   python main.py generate-kdoc
   ```

5. **Legacy compatibility test**:
   ```bash
   python TestCaseGenerator.py
   python KdocGenerator.py
   ```

### Expected Output

- Test files in `output-test/` directory
- Structured console logs showing processing progress
- Error handling for missing dependencies
- Automatic fallback to simple indexer when needed

## 🛠️ Advanced Features

### Semantic Similarity

The system uses Microsoft CodeBERT to find similar test patterns:

1. **Indexes existing tests** in `src/testcase--datastore/`
2. **Finds similar patterns** for new code
3. **Generates contextually relevant tests**

### Intelligent Prompting

- **Context-aware prompts** based on code structure
- **Existing test patterns** for consistency
- **Error handling scenarios** automatically included

### Robust Error Handling

- **Graceful degradation** when advanced features unavailable
- **Fallback mechanisms** for offline usage
- **Detailed logging** for debugging

## � Documentation

### Available Documentation

- **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)** - System architecture details
- **[docs/API.md](./docs/API.md)** - Component API documentation  
- **[docs/DIAGRAMS.md](./docs/DIAGRAMS.md)** - Visual system diagrams

### For Developers

- **Component APIs**: Each module has comprehensive docstrings
- **Type Hints**: Full type annotation throughout codebase
- **Error Handling**: Comprehensive exception handling patterns

## 🔍 Troubleshooting

### Common Issues

1. **"Import errors"**: Install dependencies with `pip install -r requirements.txt`
2. **"Ollama not responding"**: Ensure Ollama server is running with `ollama serve`
3. **"Model not found"**: Download CodeLlama with `ollama pull codellama:instruct`
4. **"Empty output"**: Check if input Kotlin files exist in `src/input-src/`
5. **"Configuration errors"**: Run `python main.py health-check` to verify setup

### Debug Mode

Enable verbose logging by setting:
```bash
export DEBUG=1
python main.py generate-tests
```

Or use the built-in logging configuration:
```bash
python main.py generate-tests --log-level DEBUG
```

### Service Status

Check individual service status:
```bash
# Health check with detailed output
python main.py health-check

# Test LLM service
python main.py test-llm

# Test embedding service
python main.py test-embedding
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment**:
   ```bash
   export OLLAMA_API_URL="http://127.0.0.1:11434/api/generate"
   export MODEL_NAME="codellama:instruct"
   ```
4. **Run health check**:
   ```bash
   python main.py health-check
   ```
5. **Run tests**:
   ```bash
   python main.py generate-tests
   ```

### Development Guidelines

- **Use the modular architecture**: Add new features through the service layer
- **Follow interfaces**: Implement abstract base classes for consistency
- **Add logging**: Use the structured logging system (`src/utils/logging.py`)
- **Configuration**: Add new settings to `src/config/settings.py`
- **Error handling**: Implement graceful degradation and fallback mechanisms

### Adding New Features

1. **Services**: Add new services to `src/services/`
2. **Core logic**: Add business logic to `src/core/`
3. **Models**: Add data models to `src/models/`
4. **Configuration**: Update `src/config/settings.py`
5. **CLI**: Add new commands to `main.py`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 External Resources

### AI/ML Resources
- **[Ollama Documentation](https://ollama.com/docs)** - Local LLM deployment guide
- **[CodeLlama Model](https://huggingface.co/codellama)** - Model specifications and usage
- **[Microsoft CodeBERT](https://huggingface.co/microsoft/codebert-base)** - Embedding model
- **[FAISS Documentation](https://faiss.ai/)** - Vector similarity search library
- **[Transformers Library](https://huggingface.co/transformers/)** - ML model framework

### Development Resources
- **[JUnit 5 Documentation](https://junit.org/junit5/docs/current/user-guide/)** - Testing framework
- **[MockK Documentation](https://mockk.io/)** - Kotlin mocking library
- **[Kotlin Documentation](https://kotlinlang.org/docs/)** - Kotlin language guide

## 🚀 Next Steps

1. **Run the system** with the new CLI:
   ```bash
   python main.py health-check
   python main.py generate-tests
   ```
2. **Examine generated tests** in `output-test/`
3. **Explore the modular architecture** in `src/`
4. **Customize configuration** in `src/config/settings.py`
5. **Add more test examples** to `src/testcase--datastore/` for improved context
6. **Explore documentation** in `docs/` for advanced features
7. **Try the legacy compatibility** with `TestCaseGenerator.py` and `KdocGenerator.py`

---

**Version**: 2.0 (Modular Architecture)  
**Last Updated**: July 2025  
**Maintained By**: Development Team  

For questions or issues, please check the troubleshooting section or review the detailed documentation in the `docs/` directory.
