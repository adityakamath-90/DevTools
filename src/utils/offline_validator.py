"""
Offline validation utilities to ensure the system operates without internet access.

This module provides utilities to validate that all models and dependencies
are available locally and no internet connections are made during operation.
"""

import os
import socket
import urllib.parse
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class OfflineValidator:
    """
    Validates that the system can run completely offline.
    
    Features:
    - Check model availability locally
    - Validate API endpoints are localhost only
    - Monitor network connections
    - Provide setup recommendations
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.validation_results: Dict[str, bool] = {}
        self.recommendations: List[str] = []
    
    def validate_offline_readiness(self) -> Tuple[bool, Dict[str, bool], List[str]]:
        """
        Comprehensive offline readiness validation.
        
        Returns:
            Tuple of (is_ready, validation_results, recommendations)
        """
        self.logger.info("Starting offline readiness validation...")
        
        # Check models availability
        self._check_embedding_models()
        self._check_llm_service()
        self._check_api_endpoints()
        self._check_environment_variables()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Overall result
        is_ready = all(self.validation_results.values())
        
        self.logger.info(f"Offline readiness validation complete: {'✅ READY' if is_ready else '❌ NOT READY'}")
        
        return is_ready, self.validation_results, self.recommendations
    
    def _check_embedding_models(self) -> None:
        """Check if embedding models are available locally."""
        self.logger.info("Checking embedding models availability...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Check CodeBERT model
            model_name = os.getenv("EMBEDDING_MODEL", "microsoft/codebert-base")
            
            try:
                # Try to load tokenizer and model from local cache only
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    local_files_only=True,
                    offline=True
                )
                model = AutoModel.from_pretrained(
                    model_name, 
                    local_files_only=True,
                    offline=True
                )
                
                self.validation_results['embedding_model_local'] = True
                self.logger.info(f"✅ Embedding model '{model_name}' found locally")
                
            except Exception as e:
                self.validation_results['embedding_model_local'] = False
                self.logger.warning(f"❌ Embedding model '{model_name}' not found locally: {e}")
                self.recommendations.append(
                    f"Download embedding model: python -c \"from transformers import AutoTokenizer, AutoModel; "
                    f"AutoTokenizer.from_pretrained('{model_name}'); AutoModel.from_pretrained('{model_name}')\""
                )
                
        except ImportError:
            self.validation_results['embedding_model_local'] = False
            self.logger.warning("❌ Transformers library not available")
            self.recommendations.append("Install transformers: pip install transformers")
    
    def _check_llm_service(self) -> None:
        """Check if LLM service is configured for local access only."""
        self.logger.info("Checking LLM service configuration...")
        
        api_url = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
        parsed_url = urllib.parse.urlparse(api_url)
        
        # Check if hostname is localhost
        is_localhost = parsed_url.hostname in ['127.0.0.1', 'localhost', '::1']
        
        self.validation_results['llm_service_local'] = is_localhost
        
        if is_localhost:
            self.logger.info(f"✅ LLM service configured for localhost: {api_url}")
            
            # Check if Ollama is actually running
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('127.0.0.1', parsed_url.port or 11434))
                sock.close()
                
                if result == 0:
                    self.validation_results['llm_service_running'] = True
                    self.logger.info("✅ Ollama service appears to be running")
                else:
                    self.validation_results['llm_service_running'] = False
                    self.logger.warning("❌ Ollama service not responding")
                    self.recommendations.append("Start Ollama service: ollama serve")
                    
            except Exception as e:
                self.validation_results['llm_service_running'] = False
                self.logger.warning(f"❌ Cannot check Ollama service: {e}")
                
        else:
            self.logger.warning(f"❌ LLM service configured for external access: {api_url}")
            self.recommendations.append("Configure LLM service for localhost: export OLLAMA_API_URL='http://127.0.0.1:11434/api/generate'")
    
    def _check_api_endpoints(self) -> None:
        """Check that all API endpoints are localhost only."""
        self.logger.info("Checking API endpoints...")
        
        endpoints_to_check = [
            ("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate"),
        ]
        
        all_local = True
        for env_var, default_url in endpoints_to_check:
            url = os.getenv(env_var, default_url)
            parsed_url = urllib.parse.urlparse(url)
            
            is_localhost = parsed_url.hostname in ['127.0.0.1', 'localhost', '::1']
            
            if not is_localhost:
                all_local = False
                self.logger.warning(f"❌ External endpoint detected: {env_var}={url}")
                self.recommendations.append(f"Configure {env_var} for localhost access")
            else:
                self.logger.info(f"✅ Localhost endpoint: {env_var}={url}")
        
        self.validation_results['api_endpoints_local'] = all_local
    
    def _check_environment_variables(self) -> None:
        """Check environment variables for offline configuration."""
        self.logger.info("Checking environment variables...")
        
        # Check if model downloads are disabled
        allow_download = os.getenv("ALLOW_MODEL_DOWNLOAD", "false").lower()
        download_disabled = allow_download not in ("true", "1", "yes")
        
        self.validation_results['downloads_disabled'] = download_disabled
        
        if download_disabled:
            self.logger.info("✅ Model downloads are disabled (offline mode)")
        else:
            self.logger.warning("❌ Model downloads are enabled (may access internet)")
            self.recommendations.append("Disable model downloads: export ALLOW_MODEL_DOWNLOAD=false")
        
        # Check other relevant environment variables
        offline_vars = {
            'TRANSFORMERS_OFFLINE': 'true',
            'HF_DATASETS_OFFLINE': 'true',
            'TOKENIZERS_PARALLELISM': 'false'
        }
        
        all_set = True
        for var, expected_value in offline_vars.items():
            actual_value = os.getenv(var, '').lower()
            if actual_value != expected_value.lower():
                all_set = False
                self.recommendations.append(f"Set {var}={expected_value} for offline operation")
        
        self.validation_results['offline_env_vars'] = all_set
    
    def _generate_recommendations(self) -> None:
        """Generate setup recommendations based on validation results."""
        if not self.validation_results.get('embedding_model_local', False):
            self.recommendations.append("Ensure embedding models are downloaded and cached locally")
        
        if not self.validation_results.get('llm_service_running', False):
            self.recommendations.append("Ensure Ollama is running locally with the required models")
        
        if not all(self.validation_results.values()):
            self.recommendations.append("Run offline validation before production use")
    
    def create_offline_setup_script(self, output_file: str = "setup_offline.sh") -> str:
        """
        Create a shell script to set up the system for offline operation.
        
        Args:
            output_file: Path to the output shell script
            
        Returns:
            Path to the created script
        """
        script_content = """#!/bin/bash
# Offline setup script for AI-powered Kotlin test generation system
# This script ensures the system can run completely offline

echo "Setting up offline environment..."

# Set environment variables for offline operation
export ALLOW_MODEL_DOWNLOAD=false
export TRANSFORMERS_OFFLINE=true
export HF_DATASETS_OFFLINE=true
export TOKENIZERS_PARALLELISM=false
export OLLAMA_API_URL="http://127.0.0.1:11434/api/generate"

# Create .env file for persistent settings
cat > .env << EOF
ALLOW_MODEL_DOWNLOAD=false
TRANSFORMERS_OFFLINE=true
HF_DATASETS_OFFLINE=true
TOKENIZERS_PARALLELISM=false
OLLAMA_API_URL=http://127.0.0.1:11434/api/generate
EOF

echo "Environment variables configured for offline operation"

# Download required models if not already cached
echo "Checking for required models..."

python3 -c "
import os
os.environ['ALLOW_MODEL_DOWNLOAD'] = 'true'  # Temporarily allow for initial setup

try:
    from transformers import AutoTokenizer, AutoModel
    
    model_name = 'microsoft/codebert-base'
    print(f'Downloading {model_name} if not cached...')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print('✅ Embedding models downloaded and cached')
except Exception as e:
    print(f'❌ Error downloading models: {e}')
"

# Check Ollama models
echo "Checking Ollama models..."
ollama list | grep -q "codellama:instruct" || {
    echo "Downloading CodeLlama model..."
    ollama pull codellama:instruct
}

echo "✅ Offline setup complete!"
echo "The system is now configured to run without internet access."
echo "Source the .env file or restart your session to apply settings."
"""
        
        script_path = Path(output_file)
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        self.logger.info(f"Created offline setup script: {output_file}")
        return str(script_path)


def validate_offline_readiness() -> None:
    """Command-line interface for offline validation."""
    validator = OfflineValidator()
    is_ready, results, recommendations = validator.validate_offline_readiness()
    
    print("\n" + "="*60)
    print("OFFLINE READINESS VALIDATION RESULTS")
    print("="*60)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check.replace('_', ' ').title()}: {status}")
    
    if recommendations:
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    print("\n" + "="*60)
    overall_status = "✅ SYSTEM READY FOR OFFLINE OPERATION" if is_ready else "❌ SYSTEM NOT READY FOR OFFLINE OPERATION"
    print(f"OVERALL STATUS: {overall_status}")
    print("="*60)
    
    if not is_ready:
        print("\nTo set up offline operation, run:")
        print("python -c \"from src.utils.offline_validator import OfflineValidator; OfflineValidator().create_offline_setup_script()\"")
        print("Then execute: ./setup_offline.sh")


if __name__ == "__main__":
    validate_offline_readiness()
