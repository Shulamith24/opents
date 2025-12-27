#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Constrained Decoding for Label-Only Generation.

This module provides a LogitsProcessor that restricts the model's output
to only generate tokens from a predefined label set (e.g., A, B, C, D, E).
This is used for classification tasks where we want to force the model
to output exactly one label token.
"""

from typing import List, Set, Union
import torch
from transformers import LogitsProcessor, PreTrainedTokenizer


class LabelConstrainedLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that masks all tokens except those in the allowed label set.
    
    This forces the model to only generate tokens that correspond to valid class labels,
    preventing it from generating explanations or other text.
    
    Example:
        >>> processor = LabelConstrainedLogitsProcessor(
        ...     allowed_labels=["A", "B", "C", "D", "E"],
        ...     tokenizer=model.tokenizer
        ... )
        >>> outputs = model.generate(
        ...     inputs_embeds=embeds,
        ...     logits_processor=[processor],
        ...     max_new_tokens=1
        ... )
    """
    
    def __init__(
        self,
        allowed_labels: List[str],
        tokenizer: PreTrainedTokenizer,
        include_eos: bool = True,
    ):
        """
        Initialize the constrained logits processor.
        
        Args:
            allowed_labels: List of allowed label strings (e.g., ["A", "B", "C"])
            tokenizer: The model's tokenizer to convert labels to token IDs
            include_eos: Whether to also allow the EOS token
        """
        super().__init__()
        
        # Convert labels to token IDs
        self.allowed_token_ids: Set[int] = set()
        
        for label in allowed_labels:
            # Get token IDs for this label (may be multiple ways to tokenize)
            # Try both with and without leading space
            for variant in [label, f" {label}", f"{label} "]:
                token_ids = tokenizer.encode(variant, add_special_tokens=False)
                # We only want single-token labels
                if len(token_ids) == 1:
                    self.allowed_token_ids.add(token_ids[0])
        
        # Optionally include EOS token
        if include_eos and tokenizer.eos_token_id is not None:
            self.allowed_token_ids.add(tokenizer.eos_token_id)
        
        # Store for debugging
        self.allowed_labels = allowed_labels
        self.tokenizer = tokenizer
        
        if len(self.allowed_token_ids) == 0:
            raise ValueError(
                f"No valid token IDs found for labels: {allowed_labels}. "
                f"Please check that the labels can be tokenized as single tokens."
            )
        
        print(f"🔒 LabelConstrainedLogitsProcessor initialized:")
        print(f"   Allowed labels: {allowed_labels}")
        print(f"   Allowed token IDs: {sorted(self.allowed_token_ids)}")
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Apply the logit mask to restrict generation to allowed tokens.
        
        Args:
            input_ids: Current generated token IDs [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
        
        Returns:
            Modified scores with non-allowed tokens set to -inf
        """
        # Create mask: -inf for all tokens not in allowed set
        mask = torch.full_like(scores, float("-inf"))
        
        # Set allowed tokens to 0 (no penalty)
        for token_id in self.allowed_token_ids:
            if token_id < scores.size(-1):
                mask[:, token_id] = 0.0
        
        # Apply mask (additive)
        return scores + mask


def get_label_token_ids(
    labels: List[str],
    tokenizer: PreTrainedTokenizer,
) -> List[int]:
    """
    Get token IDs for a list of labels.
    
    Args:
        labels: List of label strings
        tokenizer: The tokenizer to use
    
    Returns:
        List of token IDs corresponding to the labels
    """
    token_ids = []
    for label in labels:
        # Try different variants
        for variant in [label, f" {label}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
                break
    return token_ids


if __name__ == "__main__":
    # Test the constrained decoding
    from transformers import AutoTokenizer
    
    print("Testing LabelConstrainedLogitsProcessor...")
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Test with alphabet labels
    labels = ["A", "B", "C", "D", "E"]
    processor = LabelConstrainedLogitsProcessor(
        allowed_labels=labels,
        tokenizer=tokenizer,
        include_eos=True,
    )
    
    # Simulate logits
    batch_size = 2
    vocab_size = tokenizer.vocab_size
    fake_logits = torch.randn(batch_size, vocab_size)
    fake_input_ids = torch.zeros(batch_size, 10, dtype=torch.long)
    
    # Apply processor
    masked_logits = processor(fake_input_ids, fake_logits)
    
    # Check that non-allowed tokens are masked
    for token_id in range(vocab_size):
        if token_id in processor.allowed_token_ids:
            assert not torch.isinf(masked_logits[0, token_id]), f"Token {token_id} should not be masked"
        else:
            assert torch.isinf(masked_logits[0, token_id]) and masked_logits[0, token_id] < 0, f"Token {token_id} should be masked"
    
    print("✅ All tests passed!")
