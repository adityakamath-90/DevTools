"""
Logging utilities for the AI-powered Kotlin test generation system.
Provides structured logging with configurable levels and formatters.
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

try:
    from ..config.settings import config
except ImportError:
    # Fallback for when running from different contexts
    config = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information."""
        
        # Add timestamp
        record.timestamp = datetime.now().isoformat()
        
        # Add component information
        record.component = getattr(record, 'component', 'unknown')
        record.operation = getattr(record, 'operation', 'unknown')
        
        # Get the actual message
        record.message = record.getMessage()
        
        # Base format
        base_format = "[{timestamp}] [{levelname}] [{component}] {message}"
        
        # Add operation if present
        if hasattr(record, 'operation') and record.operation != 'unknown':
            base_format = "[{timestamp}] [{levelname}] [{component}:{operation}] {message}"
        
        # Add extra fields if present
        extra_fields = []
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage',
                          'timestamp', 'component', 'operation', 'message']:
                if not key.startswith('_'):
                    extra_fields.append(f"{key}={value}")
        
        if extra_fields:
            base_format += " | " + " | ".join(extra_fields)
        
        return base_format.format(**record.__dict__)


class ComponentLogger:
    """Logger for specific components with structured logging."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(f"kotlin_test_gen.{component_name}")
        
    def _log(self, level: int, message: str, operation: str = None, **kwargs) -> None:
        """Internal logging method."""
        extra = {
            'component': self.component_name,
            'operation': operation or 'general',
            **kwargs
        }
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, operation: str = None, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, operation, **kwargs)
    
    def warning(self, message: str, operation: str = None, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, operation, **kwargs)
    
    def error(self, message: str, operation: str = None, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, operation, **kwargs)
    
    def debug(self, message: str, operation: str = None, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, operation, **kwargs)
    
    def critical(self, message: str, operation: str = None, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, operation, **kwargs)


def setup_logging(
    log_level: str = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """Set up logging configuration for the application."""
    
    # Use config log level if not specified and config is available
    if log_level is None:
        if config and hasattr(config, 'processing'):
            log_level = config.processing.log_level
        else:
            log_level = "INFO"  # Fallback default
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger("kotlin_test_gen")
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = StructuredFormatter()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


def get_logger(component_name: str) -> ComponentLogger:
    """Get a logger for a specific component."""
    return ComponentLogger(component_name)


# Global logger instances for common components
embedding_logger = get_logger("embedding")
llm_logger = get_logger("llm")
test_generator_logger = get_logger("test_generator")
file_processor_logger = get_logger("file_processor")
code_parser_logger = get_logger("code_parser")

# Initialize logging with default configuration
setup_logging()
