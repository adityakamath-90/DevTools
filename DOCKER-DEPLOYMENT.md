# 🐳 DevTools Team Docker Deployment Guide

This guide provides comprehensive instructions for deploying DevTools using Docker containers for team collaboration.

## 📋 Overview

DevTools is packaged as Docker containers for easy team deployment:
- **devtools**: Main application container with Python environment
- **ollama**: AI model server running CodeLlama for documentation and test generation
- **Automated setup**: One-command deployment script handles everything

## 🔧 Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space minimum (20GB recommended)
- **CPU**: 4+ cores recommended
- **Internet**: Required for initial setup only

### Software Requirements
1. **Docker Desktop**: [Download and install](https://www.docker.com/products/docker-desktop)
   - Enable Docker Compose (included by default)
   - Allocate at least 8GB RAM to Docker
   - Ensure Docker daemon is running

2. **Command Line**: Terminal (macOS/Linux) or PowerShell (Windows)

## 🚀 Team Deployment Methods

### Method 1: Distribution Package (Recommended for Teams)

If you received a `devtools-team-YYYYMMDD.tar.gz` file:

```bash
# 1. Extract the package
tar -xzf devtools-team-YYYYMMDD.tar.gz
cd devtools-team-distribution

# 2. Run one-command setup
./deploy-team.sh

# 3. Add your Kotlin files
cp -r /path/to/your/kotlin/project/* ./input/

# 4. Generate documentation and tests
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
```

### Method 2: Git Repository

If cloning from repository:

```bash
# 1. Clone repository
git clone <repository-url>
cd DevTools

# 2. Run deployment script
./deploy-team.sh

# 3. Follow steps 3-4 from Method 1
```

## 📁 Directory Structure After Setup

```
devtools-team-distribution/
├── 📁 input/                    # Place your Kotlin files here
│   └── 📁 src/                  # Sample Kotlin files included
│       └── Calculator.kt        # Demo file for testing
├── 📁 output/                   # Generated files appear here
│   ├── 📁 kdocs/               # Generated KDoc documentation
│   └── 📁 tests/               # Generated test cases
├── 📁 data/                     # Optional: existing test cases for RAG
├── 📁 src/                      # DevTools source code
├── 📄 docker-compose-team.yml   # Docker services configuration
├── 📄 deploy-team.sh            # Automated setup script
├── 📄 Dockerfile               # Application container definition
├── 📄 README-TEAM.md           # Complete documentation
└── 📄 TEAM-SETUP.md            # Quick setup guide
```

## 🔄 Usage Workflow

### Step 1: Add Your Kotlin Files
```bash
# Copy entire Kotlin project
cp -r /path/to/your/kotlin/project/* ./input/

# Or copy specific files
cp /path/to/MyClass.kt ./input/src/
```

### Step 2: Generate Documentation and Tests
```bash
# Generate both KDoc and test cases
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both

# Or generate individually
docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc
docker-compose -f docker-compose-team.yml run --rm devtools python main.py test
```

### Step 3: Review Generated Files
```bash
# Check generated documentation
ls -la output/kdocs/

# Check generated test cases
ls -la output/tests/
```

## 🛠️ Docker Commands Reference

### Service Management
```bash
# Start services (background)
docker-compose -f docker-compose-team.yml up -d

# Stop services
docker-compose -f docker-compose-team.yml down

# View service status
docker-compose -f docker-compose-team.yml ps

# View logs
docker-compose -f docker-compose-team.yml logs -f

# Restart specific service
docker-compose -f docker-compose-team.yml restart ollama
```

### Application Usage
```bash
# Generate KDoc documentation
docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc

# Generate test cases
docker-compose -f docker-compose-team.yml run --rm devtools python main.py test

# Generate both
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both

# Get help
docker-compose -f docker-compose-team.yml run --rm devtools python main.py --help
```

### Advanced Usage
```bash
# Use custom directories
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both \
  --input-dir /app/custom-input --output-dir /app/custom-output

# Use different AI model (if available)
docker-compose -f docker-compose-team.yml run --rm \
  -e MODEL_NAME=codellama:7b devtools python main.py kdoc

# Interactive debugging
docker-compose -f docker-compose-team.yml run --rm devtools bash
```

## 🔧 Configuration

### Environment Variables
The following environment variables can be customized in `docker-compose-team.yml`:

```yaml
environment:
  - OLLAMA_API_URL=http://ollama:11434/api/generate  # AI service URL
  - MODEL_NAME=codellama:instruct                     # AI model name
  - PYTHONUNBUFFERED=1                               # Python output settings
```

### Volume Mounts
- `./input:/app/input:ro` - Your Kotlin source files (read-only)
- `./output:/app/output` - Generated documentation and tests
- `./data:/app/data:ro` - Optional existing test cases for better AI generation

## 🚨 Troubleshooting

### Common Issues

#### Docker Not Running
```bash
# Check Docker status
docker ps

# If error: "Cannot connect to Docker daemon"
# - Start Docker Desktop
# - Ensure Docker daemon is running
```

#### Memory Issues
```bash
# Check Docker memory allocation
docker system info | grep Memory

# If < 8GB allocated:
# - Open Docker Desktop Settings
# - Go to Resources > Advanced
# - Increase Memory to 8GB or more
```

#### Model Download Issues
```bash
# Check Ollama service logs
docker-compose -f docker-compose-team.yml logs ollama

# Manually pull model if needed
docker-compose -f docker-compose-team.yml exec ollama ollama pull codellama:instruct
```

#### Application Errors
```bash
# Check application logs
docker-compose -f docker-compose-team.yml logs devtools

# Test basic functionality
docker-compose -f docker-compose-team.yml run --rm devtools python -c "print('Hello from DevTools')"
```

#### Network Issues
```bash
# Check service connectivity
docker-compose -f docker-compose-team.yml exec devtools curl -f http://ollama:11434/api/tags

# Reset Docker network
docker-compose -f docker-compose-team.yml down
docker network prune
docker-compose -f docker-compose-team.yml up -d
```

### Performance Optimization

#### RAM Allocation
- Minimum: 8GB for Docker Desktop
- Recommended: 16GB+ for faster processing
- Monitor usage: `docker stats`

#### Storage Cleanup
```bash
# Remove unused Docker resources
docker system prune -a

# Remove old model data (if needed)
docker volume rm devtools_ollama_models
```

## 🔐 Air-Gap / Offline Deployment

For environments without internet access after initial setup:

1. **Pre-download models** on internet-connected machine
2. **Export Docker images** and **model data**
3. **Transfer to air-gapped environment**
4. **Import and run**

See `README-TEAM.md` for detailed air-gap instructions.

## 👥 Team Collaboration

### Sharing Results
```bash
# Package generated documentation
tar -czf team-docs-$(date +%Y%m%d).tar.gz output/

# Share with team
# Team members can extract and review
```

### Version Control
Add to `.gitignore`:
```
# Generated output (optional - may want to commit)
output/

# Docker volumes
ollama-data/
models/
```

### CI/CD Integration
```bash
# Example CI script
./deploy-team.sh
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
tar -czf documentation-$(date +%Y%m%d).tar.gz output/
```

## 📞 Support

For issues or questions:
1. Check this troubleshooting guide
2. Review logs: `docker-compose -f docker-compose-team.yml logs -f`
3. Check complete documentation in `README-TEAM.md`
4. Contact your DevTools administrator

---

**Happy documenting! 🚀**
