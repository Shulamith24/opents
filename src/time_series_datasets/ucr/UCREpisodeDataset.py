#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
UCR Episode Dataset for In-Context Learning Time Series Classification.

This module provides a dataset class that generates episodes for few-shot
classification using the UCR time series archive. Each episode contains:
- n_support samples (support set) with labels
- 1 query sample to classify

The prompt format is designed to be compatible with OpenTSLM's soft prompt architecture.
"""

import random
from typing import List, Dict, Any, Literal, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ucr_loader import load_ucr_dataset, ensure_ucr_data


# Default label alphabet (single-token labels)
LABEL_ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class UCREpisodeDataset(Dataset):
    """
    PyTorch Dataset for UCR time series classification with In-Context Learning.
    
    Each sample is an "episode" containing:
    - Support set: n_support examples with their labels
    - Query: one example to classify
    
    The dataset outputs samples in OpenTSLM's PromptWithAnswer format.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: Literal["train", "test"] = "train",
        n_support: int = 5,
        EOS_TOKEN: str = "",
        seed: Optional[int] = None,
    ):
        """
        Initialize the UCR Episode Dataset.
        
        Args:
            dataset_name: Name of the UCR dataset (e.g., "ECG5000")
            split: "train" or "test"
            n_support: Number of support examples per episode
            EOS_TOKEN: End-of-sequence token from the model's tokenizer
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.n_support = n_support
        self.EOS_TOKEN = EOS_TOKEN
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load UCR data
        ensure_ucr_data()
        train_df, test_df = load_ucr_dataset(dataset_name)
        
        # Use train split for support sampling, test for queries (or train for both during training)
        self.support_pool_df = train_df
        self.query_df = train_df if split == "train" else test_df
        
        # Get feature columns (all except "label")
        self.feature_cols = [c for c in train_df.columns if c != "label"]
        
        # Build class information
        self.classes = sorted(train_df["label"].unique().tolist())
        self.n_classes = len(self.classes)
        
        # Map original labels to single-token alphabet labels (A, B, C, ...)
        if self.n_classes > len(LABEL_ALPHABET):
            raise ValueError(f"Too many classes ({self.n_classes}). Maximum supported: {len(LABEL_ALPHABET)}")
        
        self.label_to_alpha = {cls: LABEL_ALPHABET[i] for i, cls in enumerate(self.classes)}
        self.alpha_to_label = {v: k for k, v in self.label_to_alpha.items()}
        
        # Group support pool by class for balanced sampling
        self.support_by_class = {
            cls: self.support_pool_df[self.support_pool_df["label"] == cls].reset_index(drop=True)
            for cls in self.classes
        }
        
        print(f"📊 UCREpisodeDataset initialized:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Split: {split}")
        print(f"   Classes: {self.n_classes} -> Labels: {list(self.label_to_alpha.values())[:self.n_classes]}")
        print(f"   Support pool size: {len(self.support_pool_df)}")
        print(f"   Query set size: {len(self.query_df)}")
        print(f"   Support per episode: {n_support}")
    
    def __len__(self) -> int:
        return len(self.query_df)
    
    def _normalize_series(self, series: np.ndarray) -> np.ndarray:
        """Z-score normalization for a single time series."""
        mean = series.mean()
        std = series.std()
        if std > 1e-8:
            return (series - mean) / std
        return series - mean
    
    def _sample_support_set(self, exclude_idx: Optional[int] = None) -> List[Tuple[np.ndarray, str]]:
        """
        Randomly sample n_support examples from the support pool.
        
        Returns:
            List of (normalized_series, alpha_label) tuples
        """
        support_samples = []
        
        # Strategy: sample roughly evenly from each class
        samples_per_class = max(1, self.n_support // self.n_classes)
        remaining = self.n_support - samples_per_class * self.n_classes
        
        for cls in self.classes:
            class_df = self.support_by_class[cls]
            n_samples = samples_per_class + (1 if remaining > 0 else 0)
            remaining -= 1
            
            # Sample from this class
            n_available = len(class_df)
            indices = random.sample(range(n_available), min(n_samples, n_available))
            
            for idx in indices:
                row = class_df.iloc[idx]
                series = row[self.feature_cols].astype(float).values
                series = self._normalize_series(series)
                alpha_label = self.label_to_alpha[cls]
                support_samples.append((series, alpha_label))
        
        # Shuffle support set
        random.shuffle(support_samples)
        return support_samples[:self.n_support]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an episode (support set + query) formatted for OpenTSLM.
        
        Returns:
            Dict with keys: pre_prompt, time_series, time_series_text, post_prompt, answer
        """
        # Get query sample
        query_row = self.query_df.iloc[idx]
        query_series = query_row[self.feature_cols].astype(float).values
        query_series = self._normalize_series(query_series)
        query_label = self.label_to_alpha[query_row["label"]]
        
        # Sample support set
        support_set = self._sample_support_set()
        
        # Build prompt components
        # System instruction as pre_prompt
        label_set_str = ", ".join(list(self.label_to_alpha.values())[:self.n_classes])
        pre_prompt = (
            f"You are a time series classifier. "
            f"Classify the query time series into one of these labels: [{label_set_str}]. "
            f"Output ONLY the single letter label, nothing else.\n\n"
            f"Support Examples:\n"
        )
        
        # Build time_series list and time_series_text list
        # Each support sample: text describes it, followed by the series
        time_series_list = []
        time_series_text_list = []
        
        for i, (series, label) in enumerate(support_set):
            time_series_text_list.append(f"Example {i+1}:")
            time_series_list.append(series.tolist())
            # Add the label as part of the next text segment
            if i < len(support_set) - 1:
                time_series_text_list[-1] += f"\nLabel: {label}\n"
            else:
                # Last support example
                time_series_text_list[-1] += f"\nLabel: {label}\n\nQuery:"
        
        # Add query series
        time_series_text_list.append("")  # Query has no prefix text
        time_series_list.append(query_series.tolist())
        
        # Post prompt asks for the label
        post_prompt = "\nLabel:"
        
        # Answer is just the single letter
        answer = query_label + self.EOS_TOKEN
        
        return {
            "pre_prompt": pre_prompt,
            "time_series": time_series_list,
            "time_series_text": time_series_text_list,
            "post_prompt": post_prompt,
            "answer": answer,
            # Store metadata for evaluation
            "_query_label": query_label,
            "_original_label": query_row["label"],
        }
    
    def get_label_set(self) -> List[str]:
        """Return the list of valid single-token labels."""
        return list(self.label_to_alpha.values())[:self.n_classes]
    
    def get_label_mapping(self) -> Dict[str, Any]:
        """Return the mapping from alpha labels to original class labels."""
        return self.alpha_to_label.copy()


def collate_ucr_episodes(batch: List[Dict], patch_size: int = 4) -> List[Dict]:
    """
    Collate function for UCR episodes.
    Pads time series to be multiples of patch_size.
    """
    import torch.nn.functional as F
    
    for element in batch:
        ts_list = element["time_series"]
        
        # Convert each to tensor
        ts_tensors = [torch.as_tensor(ts, dtype=torch.float32) for ts in ts_list]
        
        # Normalize each series
        normalized_tensors = []
        for ts in ts_tensors:
            mean = ts.mean()
            std = ts.std()
            if std > 1e-8:
                ts_normalized = (ts - mean) / std
            else:
                ts_normalized = ts - mean
            normalized_tensors.append(ts_normalized)
        ts_tensors = normalized_tensors
        
        # Find max length and pad to multiple of patch_size
        max_len = max(ts.size(0) for ts in ts_tensors)
        padded_len = ((max_len + patch_size - 1) // patch_size) * patch_size
        
        # Pad each series
        padded = []
        for ts in ts_tensors:
            L = ts.size(0)
            if L < padded_len:
                pad_amt = padded_len - L
                ts = F.pad(ts, (0, pad_amt), mode="constant", value=0.0)
            else:
                ts = ts[:padded_len]
            padded.append(ts)
        
        element["time_series"] = torch.stack(padded, dim=0)
    
    return batch


if __name__ == "__main__":
    # Test the dataset
    print("Testing UCREpisodeDataset...")
    
    dataset = UCREpisodeDataset(
        dataset_name="ECG5000",
        split="train",
        n_support=5,
        EOS_TOKEN="<eos>",
        seed=42,
    )
    
    # Get one sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Pre-prompt:\n{sample['pre_prompt']}")
    print(f"Number of time series: {len(sample['time_series'])}")
    print(f"Time series texts: {sample['time_series_text']}")
    print(f"Post-prompt: {sample['post_prompt']}")
    print(f"Answer: {sample['answer']}")
    print(f"Query label: {sample['_query_label']}")
    print(f"\nLabel set: {dataset.get_label_set()}")
    print(f"Label mapping: {dataset.get_label_mapping()}")
