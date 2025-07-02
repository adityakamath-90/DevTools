# DevTools - AI-Powered Kotlin Documentation & Test Generator

🤖 **One-command setup for teams** - Generate comprehensive KDoc documentation and unit tests for Kotlin projects using AI.

## 🚀 Quick Team Setup

### Prerequisites
- **Docker Desktop** installed ([Download here](https://www.docker.com/products/docker-desktop))
- **8GB RAM minimum** (16GB recommended for faster processing)
- **10GB free disk space** (for AI model storage)
- **Internet connection** (for initial AI model download - ~3.8GB)

### Setup Steps

```bash
# 1. Extract the distribution package (if you received one)
tar -xzf devtools-team-YYYYMMDD.tar.gz
cd devtools-team-distribution

# OR clone the repository
git clone <repository-url>
cd DevTools

# 2. One-command deployment (handles everything automatically)
./deploy-team.sh

# 3. Add your Kotlin files
cp -r /path/to/your/kotlin/project/* ./input/

# 4. Generate documentation and tests
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
```

**⏱️ First run**: The initial setup downloads the CodeLlama AI model (~3.8GB) and may take 10-15 minutes depending on your internet connection. Subsequent runs are much faster.

## 🐳 Docker Commands

### Core Operations
```bash
# Generate KDoc documentation only
docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc

# Generate test cases only  
docker-compose -f docker-compose-team.yml run --rm devtools python main.py test

# Generate both documentation and tests
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
```

### Service Management
```bash
# Start services in background
docker-compose -f docker-compose-team.yml up -d

# Stop all services
docker-compose -f docker-compose-team.yml down

# View service logs
docker-compose -f docker-compose-team.yml logs -f

# Check service status
docker-compose -f docker-compose-team.yml ps

# Restart services
docker-compose -f docker-compose-team.yml restart
```

### Advanced Usage
```bash
# Use custom input/output directories
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both \
  --input-dir /app/custom-input --output-dir /app/custom-output

# Run with different AI model (if available)
docker-compose -f docker-compose-team.yml run --rm -e MODEL_NAME=codellama:7b \
  devtools python main.py kdoc

# Interactive shell for debugging
docker-compose -f docker-compose-team.yml run --rm devtools bash
```

## 📁 Directory Structure

```
DevTools/
├── input/          # 👈 Place your Kotlin files here
│   └── src/        # Recommended structure for packages
├── output/         # 👈 Generated documentation and tests appear here
│   ├── kdocs/      # Generated KDoc documentation
│   └── tests/      # Generated test cases
├── data/           # Optional: existing tests for better AI generation
├── deploy-team.sh  # One-command setup script
└── docker-compose-team.yml  # Docker configuration
```

## 🔧 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB+ |
| **Storage** | 10GB free | 20GB+ free |
| **CPU** | 4 cores | 8+ cores |
| **Docker** | Latest | Latest |
| **Internet** | Required for initial setup | Optional after setup |

## 📖 Full Documentation

For complete setup instructions, troubleshooting, and advanced usage, see:
- **[Complete Team Guide](README-TEAM.md)** - Comprehensive documentation for teams
- **[Docker Deployment Guide](DOCKER-DEPLOYMENT.md)** - Detailed Docker instructions  
- **[Deployment Guide](DEPLOYMENT-GUIDE.md)** - Air-gap and offline deployment
- **[Quick Start Guide](QUICKSTART.md)** - Fast setup instructions

## 🆘 Quick Troubleshooting

```bash
# Check if services are running
docker-compose -f docker-compose-team.yml ps

# Check logs if issues occur
docker-compose -f docker-compose-team.yml logs ollama

# Restart services
docker-compose -f docker-compose-team.yml down
docker-compose -f docker-compose-team.yml up -d
```

## ✨ What It Does

- **🤖 AI-Powered KDoc**: Uses CodeLlama to generate comprehensive Kotlin documentation
- **🧪 Smart Test Generation**: Creates JUnit test cases using AI and existing patterns
- **🐳 Team-Ready**: One-command Docker deployment for entire teams
- **📁 Batch Processing**: Handles multiple Kotlin files simultaneously
- **🔒 Air-Gap Support**: Works offline after initial setup

## 🏗️ Architecture

This application consists of:
- **Main Application** (`main.py`): CLI interface for processing Kotlin files
- **Docker Services**: Containerized deployment with Ollama AI backend
- **AI Models**: CodeLlama for intelligent code analysis and generation
- **Volume Mounts**: Persistent storage for input/output and AI models

## 🔄 Development

For local development without Docker:
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OLLAMA_API_URL=http://localhost:11434/api/generate
export MODEL_NAME=codellama:instruct

# Run the application
python main.py --help
```

---

**Get your team up and running in 5 minutes! 🚀**

