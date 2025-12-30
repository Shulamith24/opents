#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
TSLANet Pretraining Script for Time Series.

This script performs MAE-style pretraining of TSLANet encoder on UCR datasets.
The pretrained weights can then be loaded for downstream ICL classification.

Usage:
    python tslanet_pretrain.py --dataset ECG200 --epochs 50

Author: OpenTSLM Team
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from model.encoder.TSLANetEncoder import TSLANetEncoder
from time_series_datasets.ucr.ucr_loader import load_ucr_dataset, UCRDataset, collate_fn


# Default hyperparameters
DEFAULT_CONFIG = {
    "batch_size": 16,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_frac": 0.1,
    "mask_ratio": 0.4,
    "grad_clip_norm": 1.0,
    "early_stop_patience": 10,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSLANet Pretraining on UCR Datasets"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, default="ECG5000",
        help="UCR dataset name (e.g., ECG200, ECG5000)"
    )
    parser.add_argument(
        "--data_path", type=str, default="./data",
        help="Path to UCR data directory"
    )
    
    # Model arguments
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size")
    parser.add_argument("--depth", type=int, default=2, help="Number of TSLANet layers")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--warmup_frac", type=float, default=DEFAULT_CONFIG["warmup_frac"])
    parser.add_argument("--mask_ratio", type=float, default=DEFAULT_CONFIG["mask_ratio"])
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_CONFIG["grad_clip_norm"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["early_stop_patience"])
    
    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="./results/tslanet_pretrain",
        help="Output directory for checkpoints"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_dir(args) -> str:
    """Create output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir,
        args.dataset,
        timestamp
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    return output_dir


def pretrain_epoch(encoder, train_loader, optimizer, scheduler, args, epoch) -> float:
    """Train for one epoch with masked reconstruction."""
    encoder.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        features, _ = batch  # Ignore labels for pretraining
        features = features.to(args.device)
        
        optimizer.zero_grad()
        
        # Forward pass with masking
        preds, targets, mask = encoder.pretrain_forward(features, mask_ratio=args.mask_ratio)
        
        # Reconstruction loss (MSE on masked positions only)
        loss = (preds - targets) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        
        # Only compute loss on masked positions
        mask = mask.bool()
        if mask.sum() > 0:
            loss = (loss * mask.float()).sum() / mask.sum()
        else:
            loss = loss.mean()
        
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(encoder.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    
    return total_loss / len(train_loader)


def validate(encoder, val_loader, args) -> float:
    """Validate and return average loss."""
    encoder.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            features, _ = batch
            features = features.to(args.device)
            
            preds, targets, mask = encoder.pretrain_forward(features, mask_ratio=args.mask_ratio)
            
            loss = (preds - targets) ** 2
            loss = loss.mean(dim=-1)
            
            mask = mask.bool()
            if mask.sum() > 0:
                loss = (loss * mask.float()).sum() / mask.sum()
            else:
                loss = loss.mean()
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def save_checkpoint(encoder, optimizer, epoch, val_loss, output_dir, filename="best_model.pt"):
    """Save encoder checkpoint."""
    checkpoint = {
        "encoder_state": encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 60)
    print("TSLANet Pretraining")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Embedding dim: {args.emb_dim}")
    print(f"Patch size: {args.patch_size}")
    print(f"Depth: {args.depth}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_output_dir(args)
    print(f"📁 Output directory: {output_dir}")
    
    # Load dataset
    print(f"\n📊 Loading {args.dataset} dataset...")
    train_df, test_df = load_ucr_dataset(args.dataset, args.data_path)
    
    # Get sequence length
    feature_cols = [c for c in train_df.columns if c != "label"]
    seq_len = len(feature_cols)
    print(f"   Sequence length: {seq_len}")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Create datasets and loaders
    train_dataset = UCRDataset(train_df)
    val_dataset = UCRDataset(test_df)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Create encoder
    print(f"\n🔧 Creating TSLANet encoder...")
    encoder = TSLANetEncoder(
        output_dim=args.emb_dim,
        dropout=args.dropout,
        patch_size=args.patch_size,
        depth=args.depth,
        max_seq_len=max(seq_len * 2, 256),  # Allow some margin
    ).to(args.device)
    
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        encoder.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\n🚀 Starting pretraining...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_checkpoint_path = None
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = pretrain_epoch(encoder, train_loader, optimizer, scheduler, args, epoch)
        print(f"Epoch {epoch} — Train loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(encoder, val_loader, args)
        print(f"Epoch {epoch} — Val loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_checkpoint_path = save_checkpoint(
                encoder, optimizer, epoch, val_loss, output_dir
            )
            print("✔️ New best model saved!")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{args.patience} epochs")
            
            if epochs_no_improve >= args.patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break
    
    # Save final model with explicit name
    final_path = os.path.join(output_dir, "tslanet_pretrained.pt")
    torch.save(encoder.state_dict(), final_path)
    print(f"\n💾 Final model saved: {final_path}")
    
    print("\n" + "=" * 60)
    print("🏁 Pretraining Complete!")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Best checkpoint: {best_checkpoint_path}")
    print(f"   Final weights: {final_path}")
    print("=" * 60)
    
    return final_path


if __name__ == "__main__":
    main()
