# 🚀 DevTools Team Quick Start

**AI-Powered Kotlin Documentation & Test Generator** - One-command Docker deployment for teams.

## ⚡ 30-Second Setup

```bash
# Extract package and run setup
tar -xzf devtools-team-YYYYMMDD.tar.gz
cd devtools-team-distribution
./deploy-team.sh

# Add your Kotlin files and generate
cp -r /path/to/your/kotlin/* ./input/
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
```

## 📋 What You Need

✅ **Docker Desktop** (latest version)  
✅ **8GB RAM** minimum (16GB recommended)  
✅ **10GB free space** (for AI model)  
✅ **Internet connection** (initial setup only)  

## 🎯 Quick Commands

```bash
# Generate KDoc documentation
docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc

# Generate test cases
docker-compose -f docker-compose-team.yml run --rm devtools python main.py test

# Generate both
docker-compose -f docker-compose-team.yml run --rm devtools python main.py both

# Stop services
docker-compose -f docker-compose-team.yml down
```

## 📁 File Structure

```
input/      ← Your Kotlin files go here
output/     ← Generated docs and tests appear here
  ├── kdocs/    ← KDoc documentation
  └── tests/    ← JUnit test cases
data/       ← Optional: existing tests for better generation
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Docker not found" | Install Docker Desktop and start it |
| "Memory error" | Allocate 8GB+ RAM to Docker in settings |
| "Model download fails" | Check internet connection, wait longer |
| "No output files" | Check logs: `docker-compose -f docker-compose-team.yml logs` |

## 📖 Complete Documentation

- **README-TEAM.md** - Full team guide with troubleshooting
- **DOCKER-DEPLOYMENT.md** - Comprehensive Docker instructions
- **Readme.md** - Project overview and commands

## 🎉 What It Does

🤖 **Smart KDoc Generation** - Uses CodeLlama AI to write comprehensive Kotlin documentation  
🧪 **Intelligent Test Creation** - Generates JUnit test cases with proper assertions  
📦 **Team-Ready Packaging** - Docker containers for easy distribution  
⚡ **Batch Processing** - Handles multiple files simultaneously  
🔒 **Air-Gap Support** - Works offline after initial setup  

---

**Get your team documenting in 5 minutes! 🚀**
