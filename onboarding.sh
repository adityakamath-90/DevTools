#!/bin/bash

# Kotlin Test Generator - Onboarding Script
# This script automates the setup process for the Kotlin Test Generator

set -e  # Exit on error

echo "🚀 Starting Kotlin Test Generator setup..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Homebrew
install_homebrew() {
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH if not already there
    if ! grep -q 'eval "$(/opt/homebrew/bin/brew shellenv)"' ~/.zprofile; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
}

# Function to install and configure Python 3.11
install_python() {
    echo "🐍 Setting up Python 3.11..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "📥 Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH if it's not already there
        if [ -x "/opt/homebrew/bin/brew" ]; then
            export PATH="/opt/homebrew/bin:$PATH"
            echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
            source ~/.zshrc
        elif [ -x "/usr/local/bin/brew" ]; then
            export PATH="/usr/local/bin:$PATH"
            echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
            source ~/.zshrc
        fi
    fi
    
    # Install Python 3.11 if not already installed
    if ! brew list python@3.11 &> /dev/null; then
        echo "📥 Installing Python 3.11 via Homebrew..."
        brew install python@3.11
    fi
    
    # Create a symlink to python3.11 if it doesn't exist
    if ! command -v python3.11 &> /dev/null; then
        echo "🔗 Creating python3.11 symlink..."
        PYTHON_PATH=$(brew --prefix python@3.11)/bin/python3.11
        if [ -f "$PYTHON_PATH" ]; then
            ln -s "$PYTHON_PATH" /usr/local/bin/python3.11
        fi
    fi
    
    # Verify Python 3.11 is available
    if command -v python3.11 &> /dev/null; then
        echo "✅ Python 3.11 is ready"
    else
        echo "❌ Failed to set up Python 3.11"
        exit 1
    fi
}

# Function to install Ollama and models
install_ollama() {
    echo "🤖 Installing Ollama..."
    if ! command_exists ollama; then
        brew install ollama
    fi
    
    # Start Ollama service
    if ! brew services list | grep -q ollama; then
        echo "🚀 Starting Ollama service..."
        brew services start ollama
    fi
    
    # Wait for Ollama to be ready
    sleep 5
    
    echo "📥 Setting up AI models..."
    
    # Pull CodeLlama model (using Ollama)
    echo "🔍 Downloading CodeLlama model (this may take several minutes)..."
    if ! ollama pull qwen2.5-coder:7b; then
        echo "❌ Failed to download CodeLlama model"
        echo "💡 Please ensure Ollama is properly installed and you have an internet connection"
        exit 1
    fi
    
    echo "✅ Successfully downloaded CodeLlama model"
}

# Function to set up the project
setup_project() {
    echo "🛠️  Setting up project..."
    
    # Remove existing virtual environment if it exists
    if [ -d "venv" ]; then
        echo "♻️  Removing existing virtual environment..."
        rm -rf venv
    fi
    
    # Create new virtual environment with Python 3.11
    echo "🐍 Creating Python 3.11 virtual environment..."
    if ! python3.11 -m venv venv; then
        echo "❌ Failed to create virtual environment with Python 3.11"
        echo "   Make sure Python 3.11 is installed and available as 'python3.11'"
        exit 1
    fi
    
    # Activate virtual environment
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
    
    # Verify we're using the correct Python
    PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    echo "   Using Python $PYTHON_VERSION in virtual environment"
    
    # Install Python dependencies with Python 3.11
    echo "📦 Installing Python dependencies (this may take a few minutes)..."
    pip install --upgrade pip setuptools wheel
    
    # First install numpy with a version that supports Python 3.11
    echo "⬇️  Installing numpy..."
    pip install --upgrade 'numpy>=1.24.0'
    
    # Install other dependencies
    echo "⬇️  Installing project dependencies..."
    pip install -r requirements.txt
    
    # Verify all dependencies are installed
    echo "✅ Dependencies installed successfully"
    
    # Download CodeBERT model (this will be cached for future use)
    echo "🔍 Pre-downloading CodeBERT model (this may take a few minutes)..."
    python -c "\
from transformers import AutoTokenizer, AutoModel
import os
print('🤖 Downloading CodeBERT model...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModel.from_pretrained('microsoft/codebert-base')
print('✅ Successfully downloaded CodeBERT model')
"
    
    echo "✅ Successfully set up all required AI models"
    echo ""
    echo "🚀 Setup complete! To start the application, run:"
    echo "   source venv/bin/activate  # Activate the virtual environment"
    echo "   streamlit run webui.py    # Start the web UI"
    
    # Create and activate virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        echo "❌ Failed to activate virtual environment"
        exit 1
    fi
    
    # Install Python dependencies
    echo "📦 Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

# Function to start the application
start_application() {
    echo "🚀 Starting Kotlin Test Generator..."
    
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "⚠️  Ollama service is not running. Starting it now..."
        brew services restart ollama
        sleep 5  # Give it time to start
    fi
    
    # Start the Streamlit app
    streamlit run src/ui/webui.py
}

# Main execution
main() {
    # Check for macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        echo "❌ This script is currently only supported on macOS"
        exit 1
    fi
    
    # Install Homebrew if not present
    if ! command_exists brew; then
        install_homebrew
    fi
    
    # Install Python
    install_python
    
    # Install Ollama and models
    install_ollama
    
    # Set up the project
    setup_project
    
    # Start the application
    start_application
}

# Run the main function
main
