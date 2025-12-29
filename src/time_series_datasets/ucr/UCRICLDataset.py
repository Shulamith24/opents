#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
UCR In-Context Learning Dataset for Time Series Classification.

This module provides an episode-based dataset for ICL-style classification,
where each sample contains:
- Support set: K-shot per class examples
- Query: The sample to classify
- Prompt template combining support + query for LLM input
"""

import os
import random
from typing import List, Literal, Tuple, Dict, Any, Optional
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset

from time_series_datasets.ucr.ucr_loader import load_ucr_dataset
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from prompt.text_prompt import TextPrompt
from prompt.prompt_with_answer import PromptWithAnswer


def int_to_label(label_int: int, unique_labels: List[int]) -> str:
    """
    Map integer label to letter label (A, B, C, ...).
    
    Args:
        label_int: Original integer label from dataset
        unique_labels: Sorted list of unique labels in dataset
    
    Returns:
        Single uppercase letter label
    """
    idx = unique_labels.index(label_int)
    return chr(ord('A') + idx)


def label_to_int(label_str: str, unique_labels: List[int]) -> int:
    """
    Map letter label back to integer label.
    
    Args:
        label_str: Single uppercase letter label
        unique_labels: Sorted list of unique labels in dataset
    
    Returns:
        Original integer label
    """
    idx = ord(label_str.upper()) - ord('A')
    return unique_labels[idx]


class UCRICLDataset(Dataset):
    """
    ICL-style UCR Dataset for time series classification.
    
    Each sample is an episode containing:
    - Support set: K-shot examples per class (randomly sampled)
    - Query: The sample to classify
    - Prompt: Formatted for LLM input with all time series embedded
    
    The dataset follows the QADataset output format for compatibility
    with OpenTSLM training pipeline.
    """
    
    # System instruction template
    SYSTEM_PROMPT = """You are a time series classifier. Analyze the support examples and classify the query.
        Output ONLY the single letter label (e.g., A, B, C, ...).

        Possible labels: {labels}

        Support Examples:"""

    # Support example template
    SUPPORT_TEMPLATE = """
        Example {idx}:
        {ts_description}
        Label: {label}"""

    # Query template
    QUERY_TEMPLATE = """
        Query:
        {ts_description}
        Label:"""

    def __init__(
        self,
        dataset_name: str = "ECG5000",
        split: Literal["train", "test"] = "train",
        k_shot: int = 1,
        EOS_TOKEN: str = "",
        raw_data_path: str = "./data",
        seed: Optional[int] = None,
    ):
        """
        Initialize the ICL dataset.
        
        Args:
            dataset_name: Name of UCR dataset (e.g., "ECG5000")
            split: "train" or "test" - determines query source
            k_shot: Number of support examples per class
            EOS_TOKEN: End-of-sequence token for answer
            raw_data_path: Path to UCR data directory
            seed: Random seed for reproducibility (None for random)
        """
        super().__init__()
        
        self.dataset_name = dataset_name
        self.split = split
        self.k_shot = k_shot
        self.EOS_TOKEN = EOS_TOKEN
        self.raw_data_path = raw_data_path
        self.seed = seed
        
        # Load data
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path)
        
        # Get unique labels (sorted for consistent mapping)
        all_labels = set(train_df["label"].tolist()) | set(test_df["label"].tolist())
        self.unique_labels = sorted(list(all_labels))
        self.num_classes = len(self.unique_labels)
        self.label_letters = [chr(ord('A') + i) for i in range(self.num_classes)]
        
        # Feature columns (exclude label)
        self.feature_cols = [c for c in train_df.columns if c != "label"]
        
        # Set query and support pools based on split
        if split == "train":
            self.query_df = train_df.reset_index(drop=True)
            self.support_df = train_df.reset_index(drop=True)  # Sample support from train
        else:
            self.query_df = test_df.reset_index(drop=True)
            self.support_df = train_df.reset_index(drop=True)  # Sample support from train
        
        # Build support pool indexed by label
        self.support_by_label: Dict[int, List[int]] = defaultdict(list)
        for idx, row in self.support_df.iterrows():
            self.support_by_label[int(row["label"])].append(idx)
        
        # Check for imbalanced classes (warn but don't fail)
        min_samples = min(len(indices) for indices in self.support_by_label.values())
        if k_shot > min_samples:
            print(f"⚠️  Warning: k_shot={k_shot} exceeds minimum samples per class ({min_samples})")
            print(f"   Some classes will use fewer than {k_shot} support examples.")
        
        # Set random state
        self.rng = random.Random(seed)
        
        print(f"📦 UCRICLDataset initialized:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Split: {split}")
        print(f"   Query samples: {len(self.query_df)}")
        print(f"   Support pool: {len(self.support_df)}")
        print(f"   Classes: {self.num_classes} ({', '.join(self.label_letters)})")
        print(f"   K-shot: {k_shot}")
    
    def __len__(self) -> int:
        return len(self.query_df)
    
    def _normalize_series(self, series: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Z-score normalize a time series.
        
        Returns:
            Normalized series, original mean, original std
        """
        mean = float(np.mean(series))
        std = float(np.std(series))
        if std < 1e-8:
            std = 1e-8
        normalized = (series - mean) / std
        return normalized, mean, std
    
    def _get_sample_data(self, df_idx: int, from_support: bool = False) -> Dict[str, Any]:
        """
        Extract normalized time series and label from dataframe row.
        
        Args:
            df_idx: Index in dataframe
            from_support: Whether to use support_df (True) or query_df (False)
        
        Returns:
            Dict with 'series', 'label_int', 'label_str', 'mean', 'std'
        """
        df = self.support_df if from_support else self.query_df
        row = df.iloc[df_idx]
        
        series = row[self.feature_cols].astype(float).values
        normalized, mean, std = self._normalize_series(series)
        label_int = int(row["label"])
        label_str = int_to_label(label_int, self.unique_labels)
        
        return {
            "series": normalized.tolist(),
            "label_int": label_int,
            "label_str": label_str,
            "mean": mean,
            "std": std,
        }
    
    def _sample_support_set(self, exclude_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Sample K-shot support examples from each class.
        
        For imbalanced datasets, if a class has fewer than k_shot samples,
        all available samples from that class will be used.
        
        Args:
            exclude_idx: Index to exclude (if query is from support pool)
        
        Returns:
            List of support sample dicts, sorted by label
        """
        support_samples = []
        
        for label in self.unique_labels:
            candidates = self.support_by_label[label].copy()
            
            # Exclude query if it's in the support pool (train split)
            if exclude_idx is not None and exclude_idx in candidates:
                candidates.remove(exclude_idx)
            
            # Sample min(k_shot, available) examples for imbalanced datasets
            num_to_sample = min(self.k_shot, len(candidates))
            
            if num_to_sample == 0:
                print(f"⚠️  Warning: No available samples for label {label}")
                continue
            
            selected = self.rng.sample(candidates, num_to_sample)
            
            for idx in selected:
                sample = self._get_sample_data(idx, from_support=True)
                support_samples.append(sample)
        
        return support_samples
    
    def _build_ts_description(self, mean: float, std: float) -> str:
        """Build time series description text."""
        return f"Time series with mean {mean:.4f} and std {std:.4f}:"
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an episode sample.
        
        Returns:
            Dict compatible with OpenTSLM training:
            - pre_prompt: System prompt + all support examples (text only)
            - time_series_text: List of TS descriptions (support + query)
            - time_series: List of normalized series tensors (support + query)
            - post_prompt: "Label:"
            - answer: Query's label letter + EOS
        """
        # Get query sample
        query = self._get_sample_data(idx, from_support=False)
        
        # Sample support set (exclude query if same pool)
        exclude_idx = idx if self.split == "train" else None
        support_samples = self._sample_support_set(exclude_idx)
        
        # Build pre_prompt with support examples (text parts)
        labels_str = ", ".join(self.label_letters)
        pre_prompt_parts = [self.SYSTEM_PROMPT.format(labels=labels_str)]
        
        # Collect all time series (support + query) for embedding
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
        
        # Add query section header to pre_prompt
        query_ts_desc = self._build_ts_description(query["mean"], query["std"])
        query_section = self.QUERY_TEMPLATE.format(ts_description=query_ts_desc)
        pre_prompt_parts.append(query_section)
        all_ts_texts.append(query_ts_desc)
        all_ts_data.append(query["series"])
        
        # Build answer
        answer = query["label_str"]
        if self.EOS_TOKEN and not answer.endswith(self.EOS_TOKEN):
            answer += self.EOS_TOKEN
        
        # Return in OpenTSLM-compatible format
        return {
            "pre_prompt": "".join(pre_prompt_parts),
            "time_series_text": all_ts_texts,
            "time_series": all_ts_data,
            "post_prompt": "",  # Query "Label:" is in pre_prompt
            "answer": answer,
            # Additional metadata for evaluation
            "query_label_int": query["label_int"],
            "query_label_str": query["label_str"],
            "num_support": len(support_samples),
        }
    
    @staticmethod
    def get_labels(num_classes: int) -> List[str]:
        """Return list of valid label letters for a given number of classes."""
        return [chr(ord('A') + i) for i in range(num_classes)]


if __name__ == "__main__":
    # Test the dataset
    print("Testing UCRICLDataset...")
    
    ds = UCRICLDataset(
        dataset_name="ECG5000",
        split="train",
        k_shot=1,
        EOS_TOKEN="",
        seed=42
    )
    
    print(f"\nDataset size: {len(ds)}")
    
    sample = ds[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"\nPre-prompt preview:\n{sample['pre_prompt'][:500]}...")
    print(f"\nNumber of time series: {len(sample['time_series'])}")
    print(f"Time series texts: {sample['time_series_text']}")
    print(f"Answer: {sample['answer']}")
    print(f"Query label: {sample['query_label_str']} (int: {sample['query_label_int']})")
