#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Constrained Decoding for Time Series Classification.

This module provides a LogitsProcessor that constrains LLM output to only
valid label tokens, preventing the model from generating invalid responses.
"""

from typing import List, Set
import torch
from transformers import LogitsProcessor, PreTrainedTokenizer


class LabelConstrainedLogitsProcessor(LogitsProcessor):
    """
    一种仅将生成内容限制在有效标签标记内的 logits 处理器。
    该处理器会将除有效标签标记之外的所有标记的 logits 设置为负无穷，从而迫使模型仅输出有效标签。
    用法：
    processor = LabelConstrainedLogitsProcessor (tokenizer, ["A", "B", "C", "D", "E"])
    outputs = model.generate (..., logits_processor=[processor]), max_new_tokens=1
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        valid_labels: List[str],
        include_eos: bool = True,
    ):
        """
        Initialize the constrained logits processor.
        
        Args:
            tokenizer: The tokenizer used by the model
            valid_labels: List of valid label strings (e.g., ["A", "B", "C"])
            include_eos: Whether to also allow EOS token (for ending generation)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.valid_labels = valid_labels
        
        # Collect all token IDs that should be allowed
        self.valid_token_ids: Set[int] = set()
        
        for label in valid_labels:
            # Try encoding with different methods to capture variants
            # Method 1: Direct encoding without special tokens
            ids = tokenizer.encode(label, add_special_tokens=False)
            self.valid_token_ids.update(ids)
            
            # Method 2: Encode with space prefix (some tokenizers need this)
            ids_with_space = tokenizer.encode(" " + label, add_special_tokens=False)
            # Only add the last token (the label itself)
            if ids_with_space:
                self.valid_token_ids.add(ids_with_space[-1])
        
        # Optionally add EOS token
        if include_eos and tokenizer.eos_token_id is not None:
            self.valid_token_ids.add(tokenizer.eos_token_id)
        
        # Also add pad token if exists (prevents issues)
        if tokenizer.pad_token_id is not None:
            self.valid_token_ids.add(tokenizer.pad_token_id)
        
        # Convert to sorted list for consistent behavior
        self.valid_token_ids_list = sorted(self.valid_token_ids)
        
        print(f"📋 LabelConstrainedLogitsProcessor initialized:")
        print(f"   Valid labels: {valid_labels}")
        print(f"   Valid token IDs: {self.valid_token_ids_list}")
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Process logits to only allow valid label tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, sequence_length]
            scores: Logits for next token [batch_size, vocab_size]
        
        Returns:
            Modified scores with invalid tokens set to -inf
        """
        # Create a mask filled with -inf
        mask = torch.full_like(scores, float('-inf'))
        
        # Set valid token positions to 0 (so adding mask doesn't change them)
        for token_id in self.valid_token_ids_list:
            if token_id < scores.shape[-1]:  # Safety check
                mask[:, token_id] = 0.0
        
        # Apply mask by adding to scores
        return scores + mask


def create_label_processor(
    tokenizer: PreTrainedTokenizer,
    num_classes: int,
    include_eos: bool = True,
) -> LabelConstrainedLogitsProcessor:
    """
    Convenience function to create a processor for A, B, C, ... labels.
    
    Args:
        tokenizer: The tokenizer used by the model
        num_classes: Number of classes (generates labels A, B, C, ...)
        include_eos: Whether to also allow EOS token
    
    Returns:
        Configured LabelConstrainedLogitsProcessor
    """
    valid_labels = [chr(ord('A') + i) for i in range(num_classes)]
    return LabelConstrainedLogitsProcessor(
        tokenizer=tokenizer,
        valid_labels=valid_labels,
        include_eos=include_eos,
    )


if __name__ == "__main__":
    # Test the processor
    print("Testing LabelConstrainedLogitsProcessor...")
    
    from transformers import AutoTokenizer
    
    # Use a small tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    valid_labels = ["A", "B", "C", "D", "E"]
    processor = LabelConstrainedLogitsProcessor(tokenizer, valid_labels)
    
    # Simulate logits
    batch_size = 2
    vocab_size = tokenizer.vocab_size
    scores = torch.randn(batch_size, vocab_size)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    # Apply processor
    masked_scores = processor(input_ids, scores)
    
    # Count non -inf values
    non_inf_count = (masked_scores > -1e9).sum(dim=-1)
    print(f"\nNon -inf tokens per sample: {non_inf_count.tolist()}")
    print(f"Expected: ~{len(processor.valid_token_ids_list)} per sample")
    
    # Verify that valid tokens have their original scores
    for token_id in processor.valid_token_ids_list[:3]:
        if token_id < vocab_size:
            orig = scores[0, token_id].item()
            masked = masked_scores[0, token_id].item()
            print(f"Token {token_id}: original={orig:.4f}, masked={masked:.4f}")
    
    print("\n✅ Test complete!")
