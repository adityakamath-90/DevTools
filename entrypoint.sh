#!/bin/sh
set -e

echo "Starting Kotlin Test Generator..."

# Create testcase-datastore directory if it doesn't exist
mkdir -p /app/testcase-datastore
chmod 777 /app/testcase-datastore

# Start Ollama in the background
echo "Starting Ollama service..."
nohup ollama serve > /var/log/ollama.log 2>&1 &

# Function to check if Ollama is ready
wait_for_ollama() {
    echo "Waiting for Ollama to be ready..."
    until curl -s http://localhost:11434/api/tags >/dev/null; do
        echo "Ollama not ready yet, waiting 5 seconds..."
        sleep 5
    done
    echo "Ollama is ready!"
}

# Download the model if it doesn't exist
echo "Checking for model codellama:instruct..."
wait_for_ollama

if ! ollama list | grep -q "codellama:instruct"; then
    echo "Downloading codellama:instruct model..."
    ollama pull codellama:instruct
    
    # Verify the model was downloaded
    if ! ollama list | grep -q "codellama:instruct"; then
        echo "Failed to download codellama:instruct model"
        exit 1
    fi
fi

# Start the web UI
echo "Starting Streamlit web UI..."
exec streamlit run src/ui/webui.py --server.port=8501 --server.address=0.0.0.0
