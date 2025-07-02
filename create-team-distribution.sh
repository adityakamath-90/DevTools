#!/bin/bash

echo "=========================================="
echo "📦 DevTools Team Distribution Creator"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create distribution directory
DIST_DIR="devtools-team-distribution"
ARCHIVE_NAME="devtools-team-$(date +%Y%m%d).tar.gz"

print_status "Creating team distribution package..."

# Remove existing distribution
rm -rf $DIST_DIR
rm -f $ARCHIVE_NAME

# Create distribution directory
mkdir -p $DIST_DIR

# Copy essential files for team deployment
print_status "Copying essential files..."

# Core application files - maintain proper structure
if [ -d "src/" ]; then
    cp -r src/ $DIST_DIR/src/
    print_status "Copied src/ directory with $(ls src/ | wc -l) files"
else
    print_warning "src/ directory not found, creating empty one"
    mkdir -p $DIST_DIR/src/
fi

# Copy main application files to root
for file in main.py requirements.txt Dockerfile docker-compose-team.yml deploy-team.sh; do
    if [ -f "$file" ]; then
        cp "$file" $DIST_DIR/
        print_status "Copied $file"
    else
        print_warning "$file not found, skipping"
    fi
done

# Documentation
for doc in README-TEAM.md Readme.md DOCKER-DEPLOYMENT.md DEPLOYMENT-GUIDE.md QUICKSTART.md LICENSE; do
    if [ -f "$doc" ]; then
        cp "$doc" $DIST_DIR/
        print_status "Copied $doc"
    else
        print_warning "$doc not found, skipping"
    fi
done

# Create empty directories for team use
mkdir -p $DIST_DIR/input/src
mkdir -p $DIST_DIR/output
mkdir -p $DIST_DIR/data

# Create sample Kotlin file
cat > $DIST_DIR/input/src/Calculator.kt << 'EOF'
package com.example

class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }
    
    fun subtract(a: Int, b: Int): Int {
        return a - b
    }
    
    fun multiply(a: Int, b: Int): Int {
        return a * b
    }
    
    fun divide(a: Int, b: Int): Double {
        if (b == 0) {
            throw IllegalArgumentException("Division by zero is not allowed")
        }
        return a.toDouble() / b.toDouble()
    }
}
EOF

# Create team setup instructions
cat > $DIST_DIR/TEAM-SETUP.md << 'EOF'
# DevTools Team Setup Instructions

## Quick Start

1. **Extract this package**
2. **Run deployment script:**
   ```bash
   ./deploy-team.sh
   ```
3. **Add your Kotlin files:**
   ```bash
   cp -r /path/to/your/kotlin/project/* ./input/
   ```
4. **Generate documentation and tests:**
   ```bash
   docker-compose -f docker-compose-team.yml run --rm devtools python main.py both
   ```

## What You Need

- **Docker Desktop** installed and running
- **8GB RAM minimum** (16GB recommended)
- **10GB free disk space** (for AI model storage)
- **Internet connection** (for initial setup only - downloads ~3.8GB AI model)

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| Storage | 10GB free | 20GB+ free |
| CPU | 4 cores | 8+ cores |
| Docker | Latest version | Latest version |

## Directory Structure

- `input/` - Place your Kotlin source files here
- `output/` - Generated documentation and tests appear here
- `data/` - Optional: existing test cases for better AI generation

## Commands

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

## 📖 Documentation

- **QUICKSTART.md** - 30-second setup guide (start here!)
- **TEAM-SETUP.md** - Quick start guide (this file)
- **README-TEAM.md** - Complete team documentation with troubleshooting
- **DOCKER-DEPLOYMENT.md** - Comprehensive Docker deployment guide
- **Readme.md** - Project overview and quick commands

## Commands Reference
EOF

# Make scripts executable
chmod +x $DIST_DIR/deploy-team.sh

# Create .gitignore for teams
cat > $DIST_DIR/.gitignore << 'EOF'
# Docker volumes and data
ollama-data/
models/

# Generated output (optional - teams may want to commit these)
# output/

# Python cache
__pycache__/
*.pyc
*.pyo

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
EOF

print_status "Creating compressed archive..."

# Create compressed archive
tar -czf $ARCHIVE_NAME $DIST_DIR/

print_success "Distribution package created: $ARCHIVE_NAME"

# Show package contents
print_status "Package contents:"
tar -tzf $ARCHIVE_NAME | head -20
echo "... (and more)"

# Show package size
PACKAGE_SIZE=$(du -h $ARCHIVE_NAME | cut -f1)
print_success "Package size: $PACKAGE_SIZE"

echo ""
echo "=========================================="
echo "📦 Distribution Package Ready!"
echo "=========================================="
echo ""
echo "📁 Archive: $ARCHIVE_NAME"
echo "📁 Size: $PACKAGE_SIZE"
echo ""
echo "🚀 Team Deployment Instructions:"
echo "1. Share $ARCHIVE_NAME with your team"
echo "2. Team members extract: tar -xzf $ARCHIVE_NAME"
echo "3. Team members run: cd $DIST_DIR && ./deploy-team.sh"
echo ""
echo "📋 What's included:"
echo "  ✅ Complete DevTools application"
echo "  ✅ Docker configuration for teams"
echo "  ✅ Automated deployment script"
echo "  ✅ Sample Kotlin files for testing"
echo "  ✅ Comprehensive documentation"
echo "  ✅ Team setup instructions"
echo ""

# Cleanup
print_warning "Cleaning up temporary directory..."
rm -rf $DIST_DIR

print_success "Distribution package ready for team deployment!"
