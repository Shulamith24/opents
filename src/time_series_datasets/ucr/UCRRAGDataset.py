#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
UCR RAG Dataset for Time Series Classification with Retrieval.

This module extends UCRICLDataset to support RAG-based support set selection.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Literal
from collections import defaultdict

from time_series_datasets.ucr.UCRICLDataset import UCRICLDataset, int_to_label
from rag.rag_index import RAGIndex


class UCRRAGDataset(UCRICLDataset):
    """
    RAG-augmented UCR Dataset for ICL classification.
    
    Extends UCRICLDataset to use retrieval for support set selection.
    
    Retrieval strategy:
    1. Compute query embedding using encoder
    2. Retrieve top-M from index (excluding query itself)
    3. Select k-shot per class from top-M by similarity
    4. Random fill if any class has fewer than k-shot
    """
    
    def __init__(
        self,
        rag_index: RAGIndex,
        encoder: torch.nn.Module,
        device: str = "cuda",
        top_m: int = 50,
        dataset_name: str = "ECG5000",
        split: Literal["train", "test"] = "train",
        k_shot: int = 1,
        EOS_TOKEN: str = "",
        raw_data_path: str = "./data",
        seed: Optional[int] = None,
    ):
        """
        Initialize RAG dataset.
        
        Args:
            rag_index: Pre-built RAG index
            encoder: TSLANet encoder for computing query embeddings
            device: Device for encoder
            top_m: Number of top candidates to retrieve
            ... (other args same as UCRICLDataset)
        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            k_shot=k_shot,
            EOS_TOKEN=EOS_TOKEN,
            raw_data_path=raw_data_path,
            seed=seed,
        )
        
        self.rag_index = rag_index
        self.encoder = encoder
        self.device = device
        self.top_m = top_m
        
        # Put encoder in eval mode
        self.encoder.eval()
        self.encoder.to(device)
        
        print(f"📊 UCRRAGDataset initialized:")
        print(f"   RAG top-M: {top_m}")
        print(f"   Index samples: {rag_index.num_samples}")
    
    def _get_query_embedding(self, idx: int) -> np.ndarray:
        """
        Compute embedding for query sample.
        
        Args:
            idx: Query index in query_df
        
        Returns:
            Normalized embedding [D]
        """
        row = self.query_df.iloc[idx]
        series = row[self.feature_cols].astype(float).values
        
        # Normalize
        mean = np.mean(series)
        std = np.std(series)
        if std < 1e-8:
            std = 1e-8
        normalized = (series - mean) / std
        
        # Convert to tensor
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            emb = self.encoder.get_embedding(x)  # [1, D]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        
        return emb.cpu().numpy().flatten()
    
    def _sample_support_set_rag(self, query_idx: int, query_emb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Sample support set using RAG retrieval.
        
        Args:
            query_idx: Query index (to exclude from retrieval)
            query_emb: Query embedding
        
        Returns:
            List of support sample dicts
        """
        # Retrieve top-M (exclude query)
        exclude_ids = [query_idx] if self.split == "train" else []
        indices, scores = self.rag_index.search(query_emb, self.top_m, exclude_ids=exclude_ids)
        
        # Select k-shot per class from retrieved results
        support_samples = []
        class_counts = defaultdict(int)
        indices_used = set()
        
        # First pass: select from retrieved results by similarity order
        for idx in indices:
            # Filter invalid indices (FAISS returns -1 for invalid)
            if idx < 0:
                continue
            
            sample = self.rag_index.get_sample(int(idx))
            label = sample["label_int"]
            
            # Skip labels not in our unique_labels (shouldn't happen, but safety check)
            if label not in self.unique_labels:
                continue
            
            label_str = int_to_label(label, self.unique_labels)
            
            if class_counts[label] < self.k_shot:
                sample["label_str"] = label_str
                support_samples.append(sample)
                class_counts[label] += 1
                indices_used.add(int(idx))
        
        # Second pass: random fill for classes with insufficient samples
        for label in self.unique_labels:
            while class_counts[label] < self.k_shot:
                # Find candidates for this label not yet used
                candidates = [
                    i for i in range(self.rag_index.num_samples)
                    if self.rag_index.support_labels[i] == label
                    and i not in indices_used
                    and i != query_idx
                ]
                
                if not candidates:
                    # No more candidates, use from retrieved if available
                    print(f"⚠️ Warning: No candidates left for label {label}")
                    break
                
                # Random select
                selected_idx = self.rng.choice(candidates)
                sample = self.rag_index.get_sample(selected_idx)
                sample["label_str"] = int_to_label(label, self.unique_labels)
                support_samples.append(sample)
                class_counts[label] += 1
                indices_used.add(selected_idx)
        
        return support_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an episode sample with RAG-retrieved support.
        
        Returns:
            Dict compatible with OpenTSLM training
        """
        # Get query sample
        query = self._get_sample_data(idx, from_support=False)
        
        # Compute query embedding
        query_emb = self._get_query_embedding(idx)
        
        # Get RAG support set
        support_samples = self._sample_support_set_rag(idx, query_emb)
        
        # Build pre_prompt with support examples
        labels_str = ", ".join(self.label_letters)
        pre_prompt_parts = [self.SYSTEM_PROMPT.format(labels=labels_str)]
        
        # Collect all time series
        all_ts_texts: List[str] = []
        all_ts_data: List[List[float]] = []
        
        # Add support examples
        for i, sample in enumerate(support_samples):
            ts_desc = self._build_ts_description(sample["mean"], sample["std"])
            example_text = self.SUPPORT_TEMPLATE.format(
                idx=i + 1,
                ts_description=ts_desc,
                label=sample["label_str"]
            )
            pre_prompt_parts.append(example_text)
            all_ts_texts.append(ts_desc)
            all_ts_data.append(sample["series"])
        
        # Add query
        query_ts_desc = self._build_ts_description(query["mean"], query["std"])
        query_section = self.QUERY_TEMPLATE.format(ts_description=query_ts_desc)
        pre_prompt_parts.append(query_section)
        all_ts_texts.append(query_ts_desc)
        all_ts_data.append(query["series"])
        
        # Build answer
        answer = query["label_str"]
        if self.EOS_TOKEN and not answer.endswith(self.EOS_TOKEN):
            answer += self.EOS_TOKEN
        
        return {
            "pre_prompt": "".join(pre_prompt_parts),
            "time_series_text": all_ts_texts,
            "time_series": all_ts_data,
            "post_prompt": "",
            "answer": answer,
            "query_label_int": query["label_int"],
            "query_label_str": query["label_str"],
            "num_support": len(support_samples),
        }


if __name__ == "__main__":
    print("UCRRAGDataset is ready for use.")
    print("Requires: RAGIndex and TSLANetEncoder")
