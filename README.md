# AI-Powered Kotlin Test Generation System

A comprehensive system that leverages advanced semantic similarity matching and Large Language Models to automatically generate high-quality JUnit 5 test cases for Kotlin code.

## 🔍 Project Overview

This system combines:
- **🤖 AI-Powered Generation**: Uses CodeLlama via Ollama for intelligent test creation
- **🧠 Semantic Similarity**: Leverages Microsoft CodeBERT for context-aware test patterns
- **⚡ Fast Processing**: FAISS indexing for efficient similarity search
- **✨ Clean Output**: Markdown-free, production-ready test code
- **📚 Documentation Generation**: Automatic KDoc comment generation for Kotlin code

## 🏗️ System Architecture

### Core Components

1. **TestCaseGenerator**: Main orchestrator that processes Kotlin files and generates tests
2. **EmbeddingIndexer**: Uses Microsoft CodeBERT model for semantic similarity matching
3. **LLMClient**: Interface to Ollama/CodeLlama for test generation
4. **PromptBuilder**: Context-aware prompt construction for better test quality
5. **KdocGenerator**: Automatic KDoc comment generation for Kotlin classes

### Data Flow

```
Kotlin Source Files → Class Detection → Similarity Search → LLM Generation → Test Files
                                    ↓
                            Existing Test Database (FAISS Index)
```

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

#### Generate Test Cases

```bash
# Generate tests for all Kotlin files in src/input-src/
python src/TestCaseGenerator.py

# Generate tests for a specific directory
python src/TestCaseGenerator.py path/to/your/kotlin/files
```

#### Generate KDoc Comments

```bash
# Add KDoc comments to Kotlin files
python src/KdocGenerator.py
```

### Output

- **Generated Tests**: Saved in `output-test/` directory
- **Test Files**: Named as `{ClassName}Test.kt`
- **Format**: JUnit 5 compatible Kotlin test code
- **Dependencies**: Uses MockK for mocking when needed

## 🛠️ System Features

### ✅ Current Capabilities

- **Advanced Class Detection**: Smart regex patterns with comment filtering
- **Semantic Similarity**: Microsoft CodeBERT embeddings for context matching
- **Test Generation**: Comprehensive JUnit 5 test creation
- **Code Cleaning**: Automatic markdown removal and formatting
- **Batch Processing**: Process entire directories of Kotlin files
- **Fallback Support**: Simple indexer when advanced features unavailable
- **Documentation**: Automatic KDoc comment generation

### 🔧 Technical Stack

- **AI Model**: CodeLlama (via Ollama)
- **Embeddings**: Microsoft CodeBERT
- **Vector Search**: FAISS
- **Language**: Python 3.9+
- **Target**: Kotlin with JUnit 5

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

1. **Run test generation**:
   ```bash
   python src/TestCaseGenerator.py
   ```

2. **Check generated tests**:
   ```bash
   ls -la output-test/
   cat output-test/CalculatorTest.kt
   ```

3. **Generate documentation**:
   ```bash
   python src/KdocGenerator.py
   ```

### Expected Output

- Test files in `output-test/` directory
- Console logs showing processing progress
- Error handling for missing dependencies

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

### Debug Mode

Enable verbose logging by setting:
```bash
export DEBUG=1
python src/TestCaseGenerator.py
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run tests**:
   ```bash
   python -m pytest tests/  # If tests exist
   ```

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

1. **Run the system** with your Kotlin files
2. **Examine generated tests** in `output-test/`
3. **Customize prompts** in `PromptBuilder.py` for your needs
4. **Add more test examples** to improve context matching
5. **Explore documentation** in `docs/` for advanced features

---

**Version**: 1.0  
**Last Updated**: July 3, 2025  
**Maintained By**: Development Team  

For questions or issues, please check the troubleshooting section or review the detailed documentation in the `docs/` directory.
