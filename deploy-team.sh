#!/bin/bash

echo "=========================================="
echo "🚀 DevTools Team Deployment Script"
echo "=========================================="
echo "Setting up AI-powered Kotlin documentation and test generator"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop from https://docker.com/products/docker-desktop"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose"
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Create necessary directories
setup_directories() {
    print_status "Creating project directories..."
    
    mkdir -p input/src
    mkdir -p output
    mkdir -p data
    
    print_success "Directories created"
}

# Create sample Kotlin file if input is empty
create_sample_files() {
    if [ ! "$(ls -A input/)" ]; then
        print_status "Creating sample Kotlin file for testing..."
        cat > input/src/Calculator.kt << 'EOF'
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
        print_success "Sample Calculator.kt created in input/src/"
    fi
}

# Build and start services
start_services() {
    print_status "Building DevTools Docker image..."
    docker-compose -f docker-compose-team.yml build
    
    if [ $? -ne 0 ]; then
        print_error "Failed to build Docker image"
        exit 1
    fi
    
    print_status "Starting DevTools services..."
    print_warning "First startup will download CodeLlama model (~3.8GB) - this may take 10-15 minutes"
    
    docker-compose -f docker-compose-team.yml up -d
    
    print_status "Waiting for services to be healthy..."
    
    # Wait for ollama to be healthy
    for i in {1..20}; do
        if docker-compose -f docker-compose-team.yml ps ollama | grep -q "healthy"; then
            print_success "Ollama service is ready"
            break
        fi
        echo "Waiting for Ollama service... ($i/20)"
        sleep 15
    done
}

# Run health check
run_health_check() {
    print_status "Running system health check..."
    
    docker-compose -f docker-compose-team.yml run --rm devtools python -c "
import requests
import os

# Test Ollama connection
try:
    response = requests.get('http://ollama:11434/api/tags')
    if response.status_code == 200:
        print('✅ Ollama service is accessible')
    else:
        print('❌ Ollama service connection failed')
        exit(1)
except Exception as e:
    print(f'❌ Ollama connection error: {e}')
    exit(1)

print('✅ All systems operational!')
"
    
    if [ $? -eq 0 ]; then
        print_success "Health check passed!"
    else
        print_error "Health check failed"
        return 1
    fi
}

# Display usage instructions
show_usage() {
    echo ""
    echo "=========================================="
    echo "🎉 Setup Complete!"
    echo "=========================================="
    echo ""
    echo "📁 Directory Structure:"
    echo "   input/     - Place your Kotlin files here"
    echo "   output/    - Generated documentation and tests will appear here"
    echo "   data/      - Optional: existing test cases for better AI generation"
    echo ""
    echo "🔧 Usage Commands:"
    echo ""
    echo "Generate KDoc documentation:"
    echo "   docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc"
    echo ""
    echo "Generate test cases:"
    echo "   docker-compose -f docker-compose-team.yml run --rm devtools python main.py test"
    echo ""
    echo "Generate both documentation and tests:"
    echo "   docker-compose -f docker-compose-team.yml run --rm devtools python main.py both"
    echo ""
    echo "Stop services:"
    echo "   docker-compose -f docker-compose-team.yml down"
    echo ""
    echo "View logs:"
    echo "   docker-compose -f docker-compose-team.yml logs -f"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Copy your Kotlin files to the 'input/' directory"
    echo "2. Run the generation commands above"
    echo "3. Check the 'output/' directory for results"
    echo ""
}

# Run demo generation
run_demo() {
    if [ -f "input/src/Calculator.kt" ]; then
        print_status "Running demo with sample Calculator.kt..."
        
        print_status "Generating KDoc documentation..."
        docker-compose -f docker-compose-team.yml run --rm devtools python main.py kdoc
        
        if [ $? -eq 0 ]; then
            print_success "Demo KDoc generation completed!"
            print_status "Check output/kdocs/ for generated documentation"
        else
            print_warning "Demo generation had issues, but setup is complete"
        fi
    fi
}

# Main execution
main() {
    echo "Starting DevTools team deployment..."
    echo ""
    
    check_docker
    setup_directories
    create_sample_files
    start_services
    
    if run_health_check; then
        run_demo
        show_usage
    else
        print_error "Setup completed but health check failed"
        print_status "Services are still starting - try running the health check again in a few minutes"
        show_usage
    fi
}

# Check if script is run with --help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "DevTools Team Deployment Script"
    echo ""
    echo "This script sets up the DevTools environment for your team."
    echo "It will:"
    echo "  1. Check Docker installation"
    echo "  2. Create necessary directories"
    echo "  3. Build and start services"
    echo "  4. Download AI model (first run only)"
    echo "  5. Run health checks"
    echo "  6. Provide usage instructions"
    echo ""
    echo "Usage: ./deploy-team.sh"
    echo ""
    exit 0
fi

# Run main function
main
