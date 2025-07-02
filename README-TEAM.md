# DevTools - AI-Powered Kotlin Documentation & Test Generator

🤖 Automatically generate comprehensive KDoc documentation and unit tests for your Kotlin projects using AI. This tool is designed for easy team deployment with Docker containers.

## 🚀 Quick Start for Teams

### Option A: One-Command Team Setup (Recommended)
```bash
# 1. Clone the repository
git clone <repository-url>
cd DevTools

# 2. Run the team deployment script (handles everything automatically)
./deploy-team.sh

# 3. Copy your Kotlin files to input directory
cp -r /path/to/your/kotlin/project/* ./input/

# 4. Generate documentation and tests
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
```

### Option B: Manual Docker Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd DevTools

# 2. Start services manually
docker-compose -f docker-compose-team.yml up -d

# 3. Wait for model download (first time only)
docker-compose -f docker-compose-team.yml logs -f ollama

# 4. Generate documentation
docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc
```

## 📋 Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Team Docker Deployment](#team-docker-deployment)
4. [Usage Guide](#usage-guide)
5. [Directory Structure](#directory-structure)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [Individual Python Setup](#individual-python-setup)

## ✨ Features

- **🤖 AI-Powered KDoc Generation**: Uses CodeLlama to generate comprehensive Kotlin documentation
- **🧪 Intelligent Test Generation**: Creates unit tests using RAG (Retrieval-Augmented Generation)
- **🐳 Team-Ready Docker**: One-command deployment for entire teams
- **🔒 Air-Gap Support**: Complete offline operation after initial setup
- **📁 Batch Processing**: Process multiple Kotlin files simultaneously
- **🔄 Smart Learning**: Learns from existing test cases to generate better tests

## 📋 Prerequisites

### For Team Docker Deployment
- **Docker Desktop 4.0+** - [Download here](https://www.docker.com/products/docker-desktop)
- **8GB RAM minimum** (16GB recommended for large projects)
- **10GB free disk space** (for AI models and containers)
- **Internet connection** (for initial model download only)

### System Requirements
- **Windows 10/11**, **macOS 10.15+**, or **Linux**
- **x86_64 architecture** (ARM64 supported but slower)

## 🐳 Team Docker Deployment

### 1. Initial Team Setup

#### Clone and Deploy
```bash
# Clone the repository
git clone <repository-url>
cd DevTools

# Run automated team deployment
./deploy-team.sh
```

**What the deployment script does:**
- ✅ Checks Docker installation
- ✅ Creates necessary directories (`input/`, `output/`, `data/`)
- ✅ Builds DevTools Docker image
- ✅ Downloads CodeLlama AI model (~3.8GB, first time only)
- ✅ Starts all services
- ✅ Runs health checks
- ✅ Creates sample files for testing

#### Manual Setup (Alternative)
```bash
# If you prefer manual control
mkdir -p input output data

# Build and start services
docker-compose -f docker-compose-team.yml build
docker-compose -f docker-compose-team.yml up -d

# Wait for initial model download (check logs)
docker-compose -f docker-compose-team.yml logs -f ollama
```

### 2. Prepare Your Kotlin Projects

```bash
# Copy your Kotlin source files to input directory
cp -r /path/to/your/kotlin/project/src/* ./input/

# Optional: Add existing test cases for better AI generation
cp -r /path/to/existing/tests/* ./data/

# Verify files are in place
ls -la input/
```

### 3. Generate Documentation and Tests

```bash
# Generate KDoc documentation only
docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc

# Generate test cases only
docker-compose -f docker-compose-team.yml run --rm devtools python main.py test

# Generate both documentation and tests
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both

# Check results
ls -la output/
```

## 📚 Usage Guide

### Basic Commands

```bash
# Start services (if not already running)
docker-compose -f docker-compose-team.yml up -d

# Generate KDoc documentation
docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc

# Generate test cases
docker-compose -f docker-compose-team.yml run --rm devtools python main.py test

# Generate both
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both

# Stop services
docker-compose -f docker-compose-team.yml down

# View logs
docker-compose -f docker-compose-team.yml logs -f

# Health check
docker-compose -f docker-compose-team.yml run --rm devtools python -c "
import requests
response = requests.get('http://ollama:11434/api/tags')
print('✅ Service healthy' if response.status_code == 200 else '❌ Service issues')
"
```

### Team Workflow

```bash
# 1. Team lead sets up the environment
./deploy-team.sh

# 2. Team members add their Kotlin projects
cp -r /my/kotlin/project/* ./input/

# 3. Generate documentation for all projects
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both

# 4. Review generated files
ls -la output/kdocs/    # Documentation
ls -la output/tests/    # Test cases

# 5. Distribute to git repository
git add output/
git commit -m "feat: Add AI-generated documentation and tests"
```

## 📁 Directory Structure

```
DevTools/
├── 📜 deploy-team.sh              # Team deployment script
├── 📜 docker-compose-team.yml     # Team Docker configuration
├── 📜 Dockerfile                  # Container definition
├── 📁 input/                      # 👈 Place your Kotlin files here
│   └── src/
│       ├── Calculator.kt          # Example file
│       └── YourKotlinFiles.kt     # Your team's files
├── 📁 output/                     # 👈 Generated files appear here
│   ├── kdocs/                     # Generated documentation
│   │   └── src/
│   │       └── Calculator.kt      # With KDoc comments
│   └── tests/                     # Generated test cases
│       └── src/
│           └── CalculatorTest.kt  # JUnit test files
├── 📁 data/                       # Optional: existing tests for better AI
│   └── ExistingTests.kt
└── 📁 src/                        # DevTools source code
    ├── main.py                    # Main application
    ├── KdocGenerator.py           # Documentation generator
    └── TestCaseGenerator.py       # Test case generator
```

## 🔧 Advanced Usage

### Custom Configuration

#### Environment Variables
```bash
# Create .env file for custom configuration
cat > .env << EOF
MODEL_NAME=codellama:instruct
OLLAMA_API_URL=http://ollama:11434/api/generate
PYTHONUNBUFFERED=1
EOF

# Use with Docker Compose
docker-compose -f docker-compose-team.yml --env-file .env up -d
```

#### Custom Input/Output Directories
```bash
# Process specific project directories
docker run --rm \
  -v /path/to/specific/kotlin/project:/app/input:ro \
  -v /path/to/custom/output:/app/output \
  devtools:latest python main.py both
```

### Batch Processing Multiple Projects

```bash
# Process multiple projects
for project in project1 project2 project3; do
  echo "Processing $project..."
  
  # Clear input directory
  rm -rf input/src/*
  
  # Copy project files
  cp -r /path/to/$project/src/* input/src/
  
  # Generate documentation and tests
  docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
  
  # Move results to project-specific directory
  mkdir -p results/$project
  cp -r output/* results/$project/
done
```

### Continuous Integration

```yaml
# .github/workflows/generate-docs.yml
name: Generate Kotlin Documentation

on:
  push:
    paths:
    - 'src/**/*.kt'

jobs:
  generate-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup DevTools
      run: |
        curl -o deploy-team.sh https://raw.githubusercontent.com/your-org/devtools/main/deploy-team.sh
        chmod +x deploy-team.sh
        ./deploy-team.sh
    
    - name: Generate Documentation
      run: |
        cp -r src/* input/
        docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
    
    - name: Commit Results
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add output/
        git commit -m "docs: Update AI-generated documentation" || exit 0
        git push
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Docker Issues
```bash
# Check Docker is running
docker --version
docker-compose --version

# Check container status
docker-compose -f docker-compose-team.yml ps

# View logs
docker-compose -f docker-compose-team.yml logs ollama
docker-compose -f docker-compose-team.yml logs devtools
```

#### 2. Model Download Issues
```bash
# Check if model is downloading
docker-compose -f docker-compose-team.yml logs -f ollama

# Manually pull model if needed
docker-compose -f docker-compose-team.yml exec ollama ollama pull codellama:instruct

# Verify model is available
docker-compose -f docker-compose-team.yml exec ollama ollama list
```

#### 3. Memory Issues
```bash
# Check Docker memory allocation
docker system info | grep -i memory

# Increase Docker memory in Docker Desktop Settings > Resources > Memory
# Recommended: 8GB minimum, 16GB for large projects
```

#### 4. No Output Generated
```bash
# Check input files exist
ls -la input/src/

# Check file permissions
ls -la output/

# Run with debug output
docker-compose -f docker-compose-team.yml run --rm \
  -e DEBUG=1 devtools python main.py kdoc
```

#### 5. Network Issues
```bash
# Test service connectivity
docker-compose -f docker-compose-team.yml run --rm devtools \
  python -c "import requests; print(requests.get('http://ollama:11434/api/tags').status_code)"

# Reset network
docker-compose -f docker-compose-team.yml down
docker network prune -f
docker-compose -f docker-compose-team.yml up -d
```

### Debug Mode
```bash
# Enable verbose logging
docker-compose -f docker-compose-team.yml run --rm \
  -e DEBUG=1 -e PYTHONUNBUFFERED=1 devtools python main.py both

# Interactive debugging
docker-compose -f docker-compose-team.yml run --rm devtools bash
# Inside container:
python main.py kdoc --help
```

## 🐍 Individual Python Setup

For developers who prefer running without Docker:

### 1. Install Dependencies
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull AI model
ollama pull codellama:instruct

# Verify service
curl http://127.0.0.1:11434/api/tags
```

### 2. Python Environment
```bash
# Install Python dependencies
pip install -r requirements.txt

# Or install individually
pip install requests sentence-transformers huggingface-hub transformers faiss-cpu numpy
```

### 3. Run Tools
```bash
# Create directories
mkdir -p input output data

# Copy Kotlin files
cp /path/to/kotlin/files/* input/

# Generate documentation
python main.py kdoc --input-dir ./input --output-dir ./output

# Generate tests
python main.py test --input-dir ./input --output-dir ./output --data-dir ./data
```

## 📖 Examples

### Example Input (Calculator.kt)
```kotlin
package com.example

class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }
    
    fun divide(a: Int, b: Int): Double {
        if (b == 0) throw IllegalArgumentException("Division by zero")
        return a.toDouble() / b.toDouble()
    }
}
```

### Generated Documentation
```kotlin
package com.example

/**
 * A calculator class that provides basic arithmetic operations.
 * 
 * This class offers fundamental mathematical operations with proper
 * error handling for edge cases like division by zero.
 */
class Calculator {
    /**
     * Adds two integers and returns the result.
     * 
     * @param a The first integer to add
     * @param b The second integer to add
     * @return The sum of a and b
     */
    fun add(a: Int, b: Int): Int {
        return a + b
    }
    
    /**
     * Divides two integers and returns the result as a double.
     * 
     * @param a The dividend (number to be divided)
     * @param b The divisor (number to divide by)
     * @return The quotient as a Double value
     * @throws IllegalArgumentException if b is zero (division by zero)
     */
    fun divide(a: Int, b: Int): Double {
        if (b == 0) throw IllegalArgumentException("Division by zero")
        return a.toDouble() / b.toDouble()
    }
}
```

### Generated Test Cases
```kotlin
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.assertThrows

class CalculatorTest {
    
    private val calculator = Calculator()
    
    @Test
    fun `test add with positive numbers`() {
        val result = calculator.add(5, 3)
        assertEquals(8, result)
    }
    
    @Test
    fun `test add with negative numbers`() {
        val result = calculator.add(-5, 3)
        assertEquals(-2, result)
    }
    
    @Test
    fun `test divide with valid numbers`() {
        val result = calculator.divide(10, 2)
        assertEquals(5.0, result, 0.001)
    }
    
    @Test
    fun `test divide by zero throws exception`() {
        assertThrows<IllegalArgumentException> {
            calculator.divide(10, 0)
        }
    }
}
```

## 🚀 Team Distribution

### Distributing to Your Team

1. **Package the repository:**
```bash
# Create distribution package
tar -czf devtools-team.tar.gz \
  --exclude='.git' \
  --exclude='output/*' \
  --exclude='ollama-data/*' \
  --exclude='models/*' \
  DevTools/
```

2. **Share with team members:**
```bash
# Each team member extracts and runs:
tar -xzf devtools-team.tar.gz
cd DevTools
./deploy-team.sh
```

3. **Alternative: Git repository:**
```bash
# Create team repository
git init
git add .
git commit -m "feat: Initial DevTools setup"
git remote add origin https://github.com/your-org/devtools.git
git push -u origin main

# Team members clone and setup:
git clone https://github.com/your-org/devtools.git
cd devtools
./deploy-team.sh
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and test with Docker
4. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 🐛 **Issues**: Open an issue on GitHub
- 💡 **Feature Requests**: Use GitHub issues with `enhancement` label  
- 📖 **Documentation**: Check this README and inline comments
- 💬 **Team Support**: Share this documentation with your team

---

**Happy coding with AI-powered documentation! 🎉🤖**

Generate better documentation and tests for your entire team with DevTools.
