import os
import glob
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingIndexer:
    def __init__(self, test_dir: str, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.test_dir = test_dir
        self.embedder = SentenceTransformer(embedding_model_name)
        self.test_cases: List[str] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        self.dimension = 0
        self._load_and_index()

    def _load_and_index(self):
        print("[INFO] Loading test cases for indexing...")
        test_files = glob.glob(os.path.join(self.test_dir, "**/*.kt"), recursive=True)
        for file in test_files:
            with open(file, "r", encoding="utf-8") as f:
                self.test_cases.append(f.read())

        if not self.test_cases:
            print("[WARN] No test cases found in directory:", self.test_dir)
            return

        print(f"[INFO] Loaded {len(self.test_cases)} test cases.")

        embeddings = self.embedder.encode(self.test_cases, convert_to_numpy=True)
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        print("[INFO] FAISS index built successfully.")

    def retrieve_similar(self, code: str, top_k: int = 3) -> List[str]:
        print("Starting retrieve_similar")

        if not self.index:
            print("Warning: Faiss index is not initialized.")
            return []

        print(f"Encoding input code: {code}")
        query_vec = self.embedder.encode([code], convert_to_numpy=True)
        print(f"Query vector shape: {query_vec.shape}")

        distances, indices = self.index.search(query_vec, top_k)
        print(f"Search distances: {distances}")
        print(f"Search indices: {indices}")

        return [self.test_cases[i] for i in indices[0] if i < len(self.test_cases)]
