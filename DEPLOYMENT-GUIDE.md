# 🚀 DevTools Docker Deployment Guide

Complete guide for deploying the DevTools Docker container from the distribution package.

## 📦 Package Contents

The `devtools-team-YYYYMMDD.tar.gz` contains:
```
devtools-team-distribution/
├── 📄 deploy-team.sh              # Automated deployment script
├── 📄 docker-compose-team.yml     # Docker services configuration
├── 📄 Dockerfile                  # Application container definition
├── 📄 main.py                     # Application entry point
├── 📄 requirements.txt            # Python dependencies
├── 📁 src/                        # Source code
├── 📁 input/                      # Your Kotlin files go here
├── 📁 output/                     # Generated files appear here
├── 📁 data/                       # Optional RAG data
└── 📖 Documentation files
```

## 🎯 Deployment Methods

### Method 1: One-Command Deployment (Recommended)

**Prerequisites:**
- Docker Desktop installed and running
- 8GB+ RAM allocated to Docker
- Internet connection (for AI model download)

**Steps:**
```bash
# 1. Extract package
tar -xzf devtools-team-20250702.tar.gz
cd devtools-team-distribution

# 2. Run automated deployment
./deploy-team.sh

# 3. Add your Kotlin files
cp -r /path/to/your/kotlin/project/* ./input/

# 4. Generate documentation and tests
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
```

**What happens during deployment:**
1. ✅ Checks Docker installation
2. ✅ Creates project directories
3. ✅ Creates sample Kotlin file for testing
4. ✅ Builds DevTools Docker image
5. ✅ Downloads Ollama and CodeLlama model (~3.8GB)
6. ✅ Starts services with health checks
7. ✅ Runs demo generation
8. ✅ Displays usage instructions

### Method 2: Manual Docker Setup

For users who prefer manual control:

```bash
# 1. Extract package
tar -xzf devtools-team-20250702.tar.gz
cd devtools-team-distribution

# 2. Create directories
mkdir -p input/src output data

# 3. Build Docker image
docker build -t devtools .

# 4. Start services
docker-compose -f docker-compose-team.yml up -d

# 5. Monitor model download (first time only)
docker-compose -f docker-compose-team.yml logs -f ollama

# 6. Wait for health check (check with)
docker-compose -f docker-compose-team.yml ps

# 7. Test the setup
docker-compose -f docker-compose-team.yml run --rm devtools python main.py --help
```

### Method 3: Step-by-Step Build

For development or customization:

```bash
# 1. Extract and examine
tar -xzf devtools-team-20250702.tar.gz
cd devtools-team-distribution
ls -la

# 2. Review configuration
cat docker-compose-team.yml
cat Dockerfile

# 3. Build application image
docker build -t my-devtools .

# 4. Start Ollama service only
docker run -d --name ollama-server \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama:latest

# 5. Download AI model
docker exec ollama-server ollama pull codellama:instruct

# 6. Run DevTools container
docker run --rm \
  --link ollama-server:ollama \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data:/app/data \
  -e OLLAMA_API_URL=http://ollama:11434/api/generate \
  my-devtools python main.py both
```

## 🌐 Network Deployment Scenarios

### Scenario 1: Internet-Connected Environment

Standard deployment with automatic model download:
```bash
tar -xzf devtools-team-20250702.tar.gz
cd devtools-team-distribution
./deploy-team.sh
```

### Scenario 2: Restricted Internet Environment

Pre-download models, then deploy:
```bash
# Download models first
docker run -d --name temp-ollama ollama/ollama:latest
docker exec temp-ollama ollama pull codellama:instruct

# Then run standard deployment
./deploy-team.sh
```

### Scenario 3: Air-Gapped Environment

Complete offline deployment:

**On Internet-Connected Machine:**
```bash
# 1. Build and save all images
tar -xzf devtools-team-20250702.tar.gz
cd devtools-team-distribution
docker build -t devtools .
docker save devtools > devtools-app.tar

# 2. Download and save Ollama with model
docker run -d --name ollama-prep ollama/ollama:latest
sleep 30
docker exec ollama-prep ollama pull codellama:instruct
docker commit ollama-prep ollama-with-model
docker save ollama-with-model > ollama-complete.tar
docker rm -f ollama-prep

# 3. Package for transfer
tar -czf air-gap-package.tar.gz \
  devtools-team-20250702.tar.gz \
  devtools-app.tar \
  ollama-complete.tar
```

**On Air-Gapped Machine:**
```bash
# 1. Extract and load images
tar -xzf air-gap-package.tar.gz
docker load < devtools-app.tar
docker load < ollama-complete.tar
docker tag ollama-with-model ollama/ollama:latest

# 2. Deploy normally
tar -xzf devtools-team-20250702.tar.gz
cd devtools-team-distribution
# Edit docker-compose-team.yml to remove model download
./deploy-team.sh
```

## 🔧 Configuration Options

### Environment Variables

Customize behavior by modifying `docker-compose-team.yml`:

```yaml
environment:
  - OLLAMA_API_URL=http://ollama:11434/api/generate
  - MODEL_NAME=codellama:instruct          # AI model to use
  - PYTHONUNBUFFERED=1                     # Python output settings
  - LANG=C.UTF-8                           # Character encoding
```

### Volume Mounts

Customize file locations:
```yaml
volumes:
  - ./input:/app/input:ro                  # Source Kotlin files
  - ./output:/app/output                   # Generated documentation/tests
  - ./data:/app/data:ro                    # Optional RAG data
  - custom_models:/app/models              # Model cache
```

### Resource Limits

For production deployment, add resource limits:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

## 📊 System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **Docker** | Latest | Latest | Docker Desktop or Engine |
| **RAM** | 8GB | 16GB+ | For AI model and processing |
| **Storage** | 10GB | 20GB+ | For models and generated files |
| **CPU** | 4 cores | 8+ cores | Faster processing |
| **Network** | 10 Mbps | 100+ Mbps | For model download |

## 🔍 Verification Steps

After deployment, verify everything works:

```bash
# 1. Check services are running
docker-compose -f docker-compose-team.yml ps

# 2. Test Ollama connectivity
docker-compose -f docker-compose-team.yml exec devtools \
  curl -f http://ollama:11434/api/tags

# 3. Test DevTools functionality
docker-compose -f docker-compose-team.yml run --rm devtools \
  python -c "import sys; sys.path.insert(0, '/app/src'); from LLMClient import LLMClient; print('✅ DevTools ready')"

# 4. Generate test documentation
echo 'class Test { fun hello() = "world" }' > input/Test.kt
docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc
ls output/kdocs/
```

## 🚨 Troubleshooting

### Common Issues

**"Docker daemon not running"**
```bash
# Start Docker Desktop
# Or start Docker service on Linux:
sudo systemctl start docker
```

**"Out of memory" during model download**
```bash
# Increase Docker memory allocation:
# Docker Desktop > Settings > Resources > Memory > 8GB+
```

**"Connection refused to Ollama"**
```bash
# Check Ollama service status
docker-compose -f docker-compose-team.yml logs ollama

# Restart services
docker-compose -f docker-compose-team.yml restart
```

**"No space left on device"**
```bash
# Clean up Docker
docker system prune -a
docker volume prune
```

## 📈 Performance Optimization

### For Faster Startup
```bash
# Pre-pull base images
docker pull python:3.11-slim
docker pull ollama/ollama:latest

# Use specific model version
# Edit docker-compose-team.yml to pin model version
```

### For Better Performance
```bash
# Allocate more resources in docker-compose-team.yml
environment:
  - OLLAMA_NUM_PARALLEL=2
  - OLLAMA_MAX_LOADED_MODELS=1
```

## 🎯 Production Deployment

For production environments:

```bash
# 1. Use production compose file
cp docker-compose-team.yml docker-compose-prod.yml

# 2. Add production settings
# - Remove debug ports
# - Add restart policies
# - Set resource limits
# - Configure logging

# 3. Deploy with production profile
docker-compose -f docker-compose-prod.yml --profile production up -d
```

---

**You're now ready to deploy DevTools Docker containers anywhere! 🚀**
