#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
RAG Index for Time Series Retrieval.

This module provides classes for building and querying embedding indices
for retrieval-augmented generation (RAG) in time series classification.

Supports:
- FAISS IndexFlatIP (cosine similarity via L2 normalization)
- Brute-force PyTorch implementation
"""

import os
import json
from typing import List, Tuple, Optional, Dict, Any, Literal
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class RAGIndex:
    """
    RAG retrieval index for time series embeddings.
    
    Supports FAISS (IndexFlatIP with L2 normalization for cosine similarity)
    and brute-force PyTorch implementation.
    
    Stores:
    - support_embs: [Ns, D] normalized embeddings
    - support_labels: [Ns] integer labels
    - support_ids: [Ns] sample indices (for exclusion)
    - support_ts: [Ns, L] raw time series data
    """
    
    def __init__(self, method: Literal["faiss", "brute"] = "faiss"):
        """
        Initialize RAG index.
        
        Args:
            method: "faiss" for FAISS IndexFlatIP, "brute" for PyTorch brute-force
        """
        if method == "faiss" and not FAISS_AVAILABLE:
            print("⚠️ FAISS not available, falling back to brute-force")
            method = "brute"
        
        self.method = method
        self.support_embs: Optional[np.ndarray] = None  # [Ns, D]
        self.support_labels: Optional[np.ndarray] = None  # [Ns]
        self.support_ids: Optional[np.ndarray] = None  # [Ns]
        self.support_ts: Optional[np.ndarray] = None  # [Ns, L]
        self.support_means: Optional[np.ndarray] = None  # [Ns]
        self.support_stds: Optional[np.ndarray] = None  # [Ns]
        
        self.faiss_index: Optional["faiss.IndexFlatIP"] = None
        self.embed_dim: int = 0
        self.num_samples: int = 0
        self.metadata: Dict[str, Any] = {}
    
    def build_from_encoder(
        self,
        encoder: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
    ):
        """
        Build index from encoder and data loader.
        
        Args:
            encoder: TSLANet encoder with get_embedding() method
            dataloader: DataLoader yielding (features, labels) tuples
            device: Device to run encoder on
        """
        encoder.eval()
        encoder.to(device)
        
        all_embs = []
        all_labels = []
        all_ids = []
        all_ts = []
        all_means = []
        all_stds = []
        
        sample_idx = 0
        
        print("📊 Building RAG index...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing embeddings"):
                features, labels = batch
                features = features.to(device)
                
                # Get embeddings using get_embedding method
                embs = encoder.get_embedding(features)  # [B, D]
                
                # L2 normalize for cosine similarity
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                
                all_embs.append(embs.cpu().numpy())
                all_labels.append(labels.numpy())
                
                # Store raw time series and stats
                for i in range(features.shape[0]):
                    ts = features[i].cpu().numpy()
                    all_ts.append(ts)
                    all_means.append(float(np.mean(ts)))
                    all_stds.append(float(np.std(ts)))
                    all_ids.append(sample_idx)
                    sample_idx += 1
        
        # Concatenate
        self.support_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
        self.support_labels = np.concatenate(all_labels, axis=0).astype(np.int64)
        self.support_ids = np.array(all_ids, dtype=np.int64)
        self.support_ts = np.stack(all_ts, axis=0).astype(np.float32)
        self.support_means = np.array(all_means, dtype=np.float32)
        self.support_stds = np.array(all_stds, dtype=np.float32)
        
        self.num_samples = self.support_embs.shape[0]
        self.embed_dim = self.support_embs.shape[1]
        
        # Build FAISS index if needed
        if self.method == "faiss":
            self._build_faiss_index()
        
        print(f"✅ RAG index built: {self.num_samples} samples, dim={self.embed_dim}")
    
    def _build_faiss_index(self):
        """Build FAISS IndexFlatIP from embeddings."""
        self.faiss_index = faiss.IndexFlatIP(self.embed_dim)
        self.faiss_index.add(self.support_embs)
    
    def search(
        self,
        query_emb: np.ndarray,
        top_m: int = 50,
        exclude_ids: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-M similar embeddings.
        
        Args:
            query_emb: Query embedding [D] or [1, D]
            top_m: Number of top results to return
            exclude_ids: Sample IDs to exclude from results
        
        Returns:
            Tuple of (indices, scores) both of shape [K] where K <= top_m
        """
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        
        # Normalize query
        query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
        query_emb = query_emb.astype(np.float32)
        
        exclude_set = set(exclude_ids) if exclude_ids else set()
        
        if self.method == "faiss":
            return self._search_faiss(query_emb, top_m, exclude_set)
        else:
            return self._search_brute(query_emb, top_m, exclude_set)
    
    def _search_faiss(
        self,
        query_emb: np.ndarray,
        top_m: int,
        exclude_set: set,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS search implementation."""
        # Search more than needed to handle exclusions
        search_k = min(top_m + len(exclude_set) + 10, self.num_samples)
        scores, indices = self.faiss_index.search(query_emb, search_k)
        
        scores = scores[0]
        indices = indices[0]
        
        # Filter excluded
        mask = np.array([idx not in exclude_set for idx in indices])
        filtered_indices = indices[mask][:top_m]
        filtered_scores = scores[mask][:top_m]
        
        return filtered_indices, filtered_scores
    
    def _search_brute(
        self,
        query_emb: np.ndarray,
        top_m: int,
        exclude_set: set,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Brute-force PyTorch search implementation."""
        # Compute cosine similarity (embeddings already normalized)
        scores = np.dot(self.support_embs, query_emb.T).flatten()
        
        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        
        # Filter excluded and take top_m
        filtered_indices = []
        filtered_scores = []
        for idx in sorted_indices:
            if idx not in exclude_set:
                filtered_indices.append(idx)
                filtered_scores.append(scores[idx])
                if len(filtered_indices) >= top_m:
                    break
        
        return np.array(filtered_indices), np.array(filtered_scores)
    
    def get_sample(self, idx: int) -> Dict[str, Any]:
        """Get sample data by index."""
        return {
            "series": self.support_ts[idx].tolist(),
            "label_int": int(self.support_labels[idx]),
            "mean": float(self.support_means[idx]),
            "std": float(self.support_stds[idx]),
            "id": int(self.support_ids[idx]),
        }
    
    def save(self, path: str):
        """Save index to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        data = {
            "method": self.method,
            "embed_dim": self.embed_dim,
            "num_samples": self.num_samples,
            "support_embs": self.support_embs,
            "support_labels": self.support_labels,
            "support_ids": self.support_ids,
            "support_ts": self.support_ts,
            "support_means": self.support_means,
            "support_stds": self.support_stds,
            "metadata": self.metadata,
        }
        
        torch.save(data, path)
        print(f"💾 RAG index saved: {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        
        self.method = data["method"]
        self.embed_dim = data["embed_dim"]
        self.num_samples = data["num_samples"]
        self.support_embs = data["support_embs"]
        self.support_labels = data["support_labels"]
        self.support_ids = data["support_ids"]
        self.support_ts = data["support_ts"]
        self.support_means = data["support_means"]
        self.support_stds = data["support_stds"]
        self.metadata = data.get("metadata", {})
        
        # Rebuild FAISS index
        if self.method == "faiss" and FAISS_AVAILABLE:
            self._build_faiss_index()
        elif self.method == "faiss" and not FAISS_AVAILABLE:
            print("⚠️ FAISS not available, using brute-force for loaded index")
            self.method = "brute"
        
        print(f"📂 RAG index loaded: {self.num_samples} samples, dim={self.embed_dim}")
    
    def get_unique_labels(self) -> List[int]:
        """Get sorted list of unique labels."""
        return sorted(list(set(self.support_labels.tolist())))


if __name__ == "__main__":
    # Test the index
    print("Testing RAGIndex...")
    
    # Create mock data
    num_samples = 100
    embed_dim = 128
    
    index = RAGIndex(method="brute")
    index.support_embs = np.random.randn(num_samples, embed_dim).astype(np.float32)
    index.support_embs /= np.linalg.norm(index.support_embs, axis=1, keepdims=True)
    index.support_labels = np.random.randint(0, 5, size=num_samples).astype(np.int64)
    index.support_ids = np.arange(num_samples).astype(np.int64)
    index.support_ts = np.random.randn(num_samples, 96).astype(np.float32)
    index.support_means = np.zeros(num_samples, dtype=np.float32)
    index.support_stds = np.ones(num_samples, dtype=np.float32)
    index.num_samples = num_samples
    index.embed_dim = embed_dim
    
    # Test search
    query = np.random.randn(embed_dim).astype(np.float32)
    indices, scores = index.search(query, top_m=10, exclude_ids=[0, 1, 2])
    
    print(f"Query shape: {query.shape}")
    print(f"Top-10 indices: {indices}")
    print(f"Top-10 scores: {scores}")
    
    # Test save/load
    test_path = "./test_rag_index.pt"
    index.save(test_path)
    
    index2 = RAGIndex()
    index2.load(test_path)
    
    print(f"Loaded index: {index2.num_samples} samples")
    
    # Cleanup
    os.remove(test_path)
    
    print("\n✅ RAGIndex test passed!")
