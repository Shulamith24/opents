#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Build RAG Index for Time Series Retrieval.

This script pre-computes embeddings for the training set and builds
a RAG index for retrieval-augmented ICL classification.

Usage:
    python build_rag_index.py --dataset ECG200 \
        --encoder_checkpoint ./results/tslanet_pretrain/ECG200/.../tslanet_pretrained.pt

Author: OpenTSLM Team
"""

import sys
import os
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from torch.utils.data import DataLoader

from model.encoder.TSLANetEncoder import TSLANetEncoder
from time_series_datasets.ucr.ucr_loader import load_ucr_dataset, UCRDataset, collate_fn
from rag.rag_index import RAGIndex


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build RAG Index for Time Series Retrieval"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, default="ECG200",
        help="UCR dataset name"
    )
    parser.add_argument(
        "--data_path", type=str, default="./data",
        help="Path to UCR data directory"
    )
    
    # Encoder arguments
    parser.add_argument(
        "--encoder_checkpoint", type=str, required=True,
        help="Path to TSLANet pretrained checkpoint"
    )
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    
    # Index arguments
    parser.add_argument(
        "--method", type=str, default="faiss",
        choices=["faiss", "brute"],
        help="Index method"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="./results/rag_index",
        help="Output directory"
    )
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Building RAG Index")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Encoder: {args.encoder_checkpoint}")
    print(f"Method: {args.method}")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📊 Loading {args.dataset} dataset...")
    train_df, test_df = load_ucr_dataset(args.dataset, args.data_path)
    
    feature_cols = [c for c in train_df.columns if c != "label"]
    seq_len = len(feature_cols)
    print(f"   Sequence length: {seq_len}")
    print(f"   Train samples: {len(train_df)}")
    
    # Create data loader for training set
    train_dataset = UCRDataset(train_df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important: keep order for ID mapping
        collate_fn=collate_fn,
    )
    
    # Load encoder
    print(f"\n🔧 Loading TSLANet encoder...")
    
    # Infer max_seq_len from checkpoint
    state_dict = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
    if "encoder_state" in state_dict:
        encoder_state = state_dict["encoder_state"]
    else:
        encoder_state = state_dict
    
    # Get max_seq_len from pos_embed
    if "pos_embed" in encoder_state:
        num_patches = encoder_state["pos_embed"].shape[1]
        stride = args.patch_size // 2
        max_seq_len = (num_patches - 1) * stride + args.patch_size
    else:
        max_seq_len = seq_len * 2
    
    encoder = TSLANetEncoder(
        output_dim=args.emb_dim,
        patch_size=args.patch_size,
        depth=args.depth,
        max_seq_len=max_seq_len,
    )
    encoder.load_state_dict(encoder_state)
    encoder.to(args.device)
    encoder.eval()
    
    print(f"   Encoder loaded, max_seq_len={max_seq_len}")
    
    # Build index
    rag_index = RAGIndex(method=args.method)
    rag_index.build_from_encoder(encoder, train_loader, device=args.device)
    
    # Add metadata
    rag_index.metadata = {
        "dataset": args.dataset,
        "encoder_checkpoint": args.encoder_checkpoint,
        "build_time": datetime.now().isoformat(),
    }
    
    # Save index
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"rag_index_{timestamp}.pt")
    rag_index.save(output_path)
    
    # Also save as "latest"
    latest_path = os.path.join(output_dir, "rag_index_latest.pt")
    rag_index.save(latest_path)
    
    print("\n" + "=" * 60)
    print("🏁 RAG Index Build Complete!")
    print(f"   Samples: {rag_index.num_samples}")
    print(f"   Embed dim: {rag_index.embed_dim}")
    print(f"   Saved to: {output_path}")
    print(f"   Latest: {latest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
