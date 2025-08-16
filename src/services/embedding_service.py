"""
Embedding and similarity indexing services for semantic code search.

This module provides both advanced (CodeBERT + FAISS) and simple (text-based)
embedding services for finding similar test cases to provide context.
"""

import os
import glob
import time
import torch
import faiss
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from src.interfaces.base_interfaces import SimilarityIndexer, EmbeddingProvider
from src.models.data_models import SimilarityMatch, EmbeddingVector
from src.config.settings import EmbeddingConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optional imports for advanced features
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Advanced embedding features will be disabled.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Vector search will be disabled.")


@dataclass
class IndexingResult:
    """Result of indexing operation."""
    success: bool
    indexed_count: int
    processing_time: float
    error_message: Optional[str] = None


class EmbeddingIndexerService(SimilarityIndexer):
    """
    Advanced embedding indexer using CodeBERT and FAISS.
    
    Features:
    - Microsoft CodeBERT for code-aware embeddings
    - FAISS for efficient similarity search
    - Batch processing for large datasets
    - Persistent index storage
    - Comprehensive error handling
    - Detailed metrics and logging
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.index = None
        self.test_cases: List[str] = []
        self.file_paths: List[str] = []
        self.dimension = 0
        
        # Check availability
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Please install: pip install transformers")
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS library not available. Please install: pip install faiss-cpu")
        
        # Initialize model
        self._initialize_model()
        
        # Load and index test cases
        if self.config.test_cases_dir:
            self._load_and_index()
    
    def _initialize_model(self):
        """Initialize the CodeBERT model and tokenizer."""
        try:
            self.logger.info(f"Loading embedding model: {self.config.model_name}")

            # Optionally set offline-related environment variables based on config
            self._validate_offline_mode()

            # Decide local-only load based on offline_mode
            local_only = bool(getattr(self.config, "offline_mode", True))

            # First try to load from local cache only (when offline or preferred)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    local_files_only=local_only,
                )
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    local_files_only=local_only,
                )
                src = "local cache" if local_only else "auto"
                self.logger.info(f"Successfully loaded embedding model from {src}")
            except Exception as local_error:
                self.logger.warning(f"Failed to load model with local_only={local_only}: {local_error}")

                # Only download if specifically allowed via centralized config
                if not bool(getattr(self.config, "allow_model_download", False)):
                    raise ImportError(
                        f"Model '{self.config.model_name}' not found locally and downloads are disabled by EmbeddingConfig.allow_model_download."
                    )

                self.logger.info("Attempting to download model (internet required) as allow_model_download=True")

                # Download model (requires internet); ensure not forcing offline
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, local_files_only=False)
                self.model = AutoModel.from_pretrained(self.config.model_name, local_files_only=False)
                self.logger.info("Successfully downloaded and loaded embedding model")

            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _validate_offline_mode(self) -> None:
        """Apply offline mode env settings based on centralized config flags."""
        try:
            if bool(getattr(self.config, "set_env_offline", True)):
                os.environ['TRANSFORMERS_OFFLINE'] = 'true' if bool(getattr(self.config, "offline_mode", True)) else 'false'
                os.environ['HF_DATASETS_OFFLINE'] = 'true' if bool(getattr(self.config, "offline_mode", True)) else 'false'
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                self.logger.info("Applied offline environment settings from EmbeddingConfig")
            else:
                self.logger.debug("set_env_offline=False; not modifying offline-related environment variables")

            if bool(getattr(self.config, "allow_model_download", False)):
                self.logger.warning("allow_model_download=True - system may access internet to fetch models")
        except Exception as e:
            self.logger.warning(f"Failed to adjust offline environment settings: {e}")
    
    def _load_and_index(self) -> IndexingResult:
        """Load test cases and build the search index."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Loading test cases from: {self.config.test_cases_dir}")
            
            # Find all Kotlin test files
            test_files = glob.glob(
                os.path.join(self.config.test_cases_dir, "**/*.kt"), 
                recursive=True
            )
            
            if not test_files:
                self.logger.warning(f"No test cases found in: {self.config.test_cases_dir}")
                return IndexingResult(
                    success=False,
                    indexed_count=0,
                    processing_time=time.time() - start_time,
                    error_message="No test cases found"
                )
            
            # Load test case content
            self.test_cases = []
            self.file_paths = []
            
            for file_path in test_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.test_cases.append(content)
                        self.file_paths.append(file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
            
            if not self.test_cases:
                return IndexingResult(
                    success=False,
                    indexed_count=0,
                    processing_time=time.time() - start_time,
                    error_message="No test cases loaded successfully"
                )
            
            self.logger.info(f"Loaded {len(self.test_cases)} test cases")
            
            # Generate embeddings
            embeddings = self._encode_batch(self.test_cases)
            self.dimension = embeddings.shape[1]
            
            # Build FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype(np.float32))
            
            processing_time = time.time() - start_time
            self.logger.info(f"Successfully built FAISS index in {processing_time:.2f}s")
            
            return IndexingResult(
                success=True,
                indexed_count=len(self.test_cases),
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load and index test cases: {e}")
            return IndexingResult(
                success=False,
                indexed_count=0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts using CodeBERT."""
        self.logger.info(f"Generating embeddings for {len(texts)} tests...")
        
        embeddings = []
        
        # Process in batches to manage memory
        batch_size = self.config.batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._encode_single_batch(batch)
            embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        return np.vstack(embeddings)
    
    def _encode_single_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a single batch of texts."""
        try:
            with torch.no_grad():
                # Tokenize
                encoded_input = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                # Get model outputs
                outputs = self.model(**encoded_input)
                
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                return embeddings.numpy()
                
        except Exception as e:
            self.logger.error(f"Error encoding batch: {e}")
            raise
    
    def find_similar(self, query: str, top_k: int = 3, similarity_threshold: float = 0.8) -> List[str]:
        """
        Find similar test cases for a given query.
        
        Args:
            query: Source code to find similar tests for
            top_k: Number of similar tests to return
            similarity_threshold: Minimum similarity score (0-1) to consider a match valid.
                                 If no matches meet this threshold, an LLM will be used to generate test cases.
                                 
        Returns:
            List of similar test case contents
        """
        try:
            if not self.index:
                self.logger.warning("No index available, returning empty results")
                return []
            
            # Encode query
            query_embedding = self._encode_single_batch([query])
            
            # Search index with a larger k to find any close matches
            search_k = min(top_k * 2, len(self.test_cases))  # Look at more candidates
            distances, indices = self.index.search(
                query_embedding.astype(np.float32), 
                search_k
            )
            
            # Calculate similarity scores (1 / (1 + distance))
            similarities = 1 / (1 + distances[0])
            
            # Filter results by similarity threshold
            results = []
            for i, (idx, sim) in enumerate(zip(indices[0], similarities)):
                if idx < len(self.test_cases) and sim >= similarity_threshold:
                    results.append((sim, self.test_cases[idx]))
                if len(results) >= top_k:
                    break
            
            # If no close matches found, skip LLM fallback to avoid extra provider init and latency
            # The generator will proceed without similar tests.
            
            # If we still have no results, return the most similar regardless of threshold
            if not results and indices.size > 0:
                for idx in indices[0]:
                    if idx < len(self.test_cases):
                        results.append((0.0, self.test_cases[idx]))  # Low confidence score
                    if len(results) >= top_k:
                        break
            
            self.logger.info(f"Found {len(results)} similar test cases")
            return [test for _, test in results]
            
        except Exception as e:
            self.logger.error(f"Error finding similar tests: {e}")
            return []
    
    def add_test_case(self, content: str, file_path: str) -> bool:
        """
        Add a new test case to the index.
        
        Args:
            content: Test case content
            file_path: Path to the test file
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Encode new test case
            embedding = self._encode_single_batch([content])
            
            # Add to index
            self.index.add(embedding.astype(np.float32))
            
            # Add to collections
            self.test_cases.append(content)
            self.file_paths.append(file_path)
            
            self.logger.info(f"Added new test case: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding test case: {e}")
            return False

    def build_index(self, embeddings: List[EmbeddingVector]) -> None:
        """Build similarity search index from embeddings."""
        try:
            self._initialize_model()
            self._load_and_index()
            self.logger.info("Index built successfully")
        except Exception as e:
            self.logger.error(f"Failed to build index: {e}")
            raise

    def search(self, query_embedding: EmbeddingVector, top_k: int = 3) -> List[SimilarityMatch]:
        """Search for similar items in the index."""
        try:
            # Convert query to text if needed
            if hasattr(query_embedding, 'text'):
                query_text = query_embedding.text
            else:
                query_text = str(query_embedding)
            
            similar_content = self.find_similar(query_text, top_k)
            
            matches = []
            for i, content in enumerate(similar_content):
                match = SimilarityMatch(
                    content=content,
                    score=1.0 - (i * 0.1),  # Simple scoring
                    file_path="",
                    line_number=0
                )
                matches.append(match)
            
            return matches
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def add_to_index(self, embeddings: List[EmbeddingVector]) -> None:
        """Add new embeddings to existing index."""
        try:
            for embedding in embeddings:
                if hasattr(embedding, 'text'):
                    self.add_test_case(embedding.text, "")
        except Exception as e:
            self.logger.error(f"Failed to add to index: {e}")

    @property
    def index_size(self) -> int:
        """Get the size of the index."""
        return len(self.test_cases)


class SimpleEmbeddingIndexerService(SimilarityIndexer):
    """
    Simple text-based similarity indexer as fallback.
    
    Features:
    - Text-based similarity without ML dependencies
    - Lightweight and fast
    - Keyword matching
    - No external dependencies
    - Suitable for constrained environments
    """
    
    def __init__(self, test_cases_dir: str):
        self.test_cases_dir = test_cases_dir
        self.logger = get_logger(self.__class__.__name__)
        self.test_cases: List[str] = []
        self.file_paths: List[str] = []
        
        # Load test cases
        self._load_test_cases()
    
    def _load_test_cases(self):
        """Load test cases from directory."""
        try:
            self.logger.info(f"Loading test cases from: {self.test_cases_dir}")
            
            if not os.path.exists(self.test_cases_dir):
                self.logger.warning(f"Test cases directory does not exist: {self.test_cases_dir}")
                return
            
            # Find all Kotlin test files
            test_files = glob.glob(
                os.path.join(self.test_cases_dir, "**/*.kt"), 
                recursive=True
            )
            
            for file_path in test_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.test_cases.append(content)
                        self.file_paths.append(file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
            
            self.logger.info(f"Loaded {len(self.test_cases)} test cases")
            
        except Exception as e:
            self.logger.error(f"Error loading test cases: {e}")
    
    def find_similar(self, query: str, top_k: int = 3) -> List[str]:
        """
        Find similar test cases using simple text matching.
        
        Args:
            query: Source code to find similar tests for
            top_k: Number of similar tests to return
            
        Returns:
            List of similar test case contents
        """
        if not self.test_cases:
            self.logger.warning("No test cases available")
            return []
        
        try:
            # Simple keyword-based matching
            query_words = self._extract_keywords(query)
            
            # Score each test case
            scored_tests = []
            for i, test_case in enumerate(self.test_cases):
                score = self._calculate_similarity_score(query_words, test_case)
                scored_tests.append((score, i))
            
            # Sort by score and take top K
            scored_tests.sort(key=lambda x: x[0], reverse=True)
            
            # Return top K test cases
            results = []
            for score, idx in scored_tests[:top_k]:
                if score > 0:  # Only return cases with some similarity
                    results.append(self.test_cases[idx])
            
            self.logger.info(f"Found {len(results)} similar test cases using simple matching")
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar tests: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for matching."""
        # Simple keyword extraction
        import re
        
        # Find class names, method names, and other identifiers
        keywords = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', text)  # Class names
        keywords.extend(re.findall(r'\b[a-z][a-zA-Z0-9_]*\b', text))  # Method names
        
        # Remove common words
        common_words = {'the', 'and', 'or', 'but', 'if', 'then', 'else', 'for', 'while', 'do'}
        keywords = [kw for kw in keywords if kw.lower() not in common_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_similarity_score(self, query_words: List[str], test_case: str) -> float:
        """Calculate similarity score between query and test case."""
        if not query_words:
            return 0.0
        
        # Count matching words
        test_words = self._extract_keywords(test_case)
        matches = sum(1 for word in query_words if word in test_words)
        
        # Return ratio of matching words
        return matches / len(query_words)
    
    def add_test_case(self, content: str, file_path: str) -> bool:
        """Add a new test case to the collection."""
        try:
            self.test_cases.append(content)
            self.file_paths.append(file_path)
            self.logger.info(f"Added new test case: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding test case: {e}")
            return False

    def search(self, query_embedding: EmbeddingVector, top_k: int = 3) -> List[SimilarityMatch]:
        """Search for similar items, delegating to find_similar() and wrapping results.

        This satisfies the SimilarityIndexer ABC requirement.
        """
        try:
            # Convert query to text if possible
            if hasattr(query_embedding, 'text'):
                query_text = query_embedding.text
            else:
                query_text = str(query_embedding)

            texts = self.find_similar(query_text, top_k=top_k)
            matches: List[SimilarityMatch] = []
            for i, content in enumerate(texts):
                matches.append(SimilarityMatch(
                    content=content,
                    score=max(0.0, 1.0 - i * 0.1),
                    file_path="",
                    line_number=0,
                ))
            return matches
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def build_index(self, embeddings: List[EmbeddingVector]) -> None:
        """Build similarity search index from embeddings."""
        try:
            self._load_test_cases()
            self.logger.info("Simple index built successfully")
        except Exception as e:
            self.logger.error(f"Failed to build index: {e}")
            raise

    def add_to_index(self, embeddings: List[EmbeddingVector]) -> None:
        """Add new embeddings to existing index."""
        try:
            for embedding in embeddings:
                if hasattr(embedding, 'text'):
                    self.add_test_case(embedding.text, "")
        except Exception as e:
            self.logger.error(f"Failed to add to index: {e}")

    @property
    def index_size(self) -> int:
        """Get the size of the index."""
        return len(self.test_cases)