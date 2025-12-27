#!/usr/bin/env python3
#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
UCR Time Series Classification Training Script (M1: Soft Prompt + Random Sampling)

This script trains OpenTSLM for time series classification on UCR datasets
using in-context learning with randomly sampled support sets.

Features:
- Loads pre-trained model from HuggingFace
- Uses UCREpisodeDataset for few-shot classification format
- Applies constrained decoding to force single-token label output
- Evaluates classification accuracy

Usage:
    python train_ucr_classification.py --dataset_name ECG5000 --n_support 5
"""

import sys
import os
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from model.llm.OpenTSLM import OpenTSLM
from model.llm.constrained_decoding import LabelConstrainedLogitsProcessor
from time_series_datasets.ucr.UCREpisodeDataset import (
    UCREpisodeDataset,
    collate_ucr_episodes,
)
from model_config import PATCH_SIZE, GRAD_CLIP_NORM


# Default configuration
DEFAULT_CONFIG = {
    "dataset_name": "ECG5000",
    "n_support": 5,
    "batch_size": 4,
    "num_epochs": 10,
    "lr_encoder": 1e-4,
    "lr_projector": 5e-5,
    "warmup_frac": 0.1,
    "weight_decay": 0.01,
    "early_stop_patience": 5,
    "eval_every_n_epochs": 1,
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train OpenTSLM for UCR time series classification"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_CONFIG["dataset_name"],
        help="UCR dataset name (e.g., ECG5000)",
    )
    parser.add_argument(
        "--n_support",
        type=int,
        default=DEFAULT_CONFIG["n_support"],
        help="Number of support examples per episode",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_CONFIG["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr_encoder",
        type=float,
        default=DEFAULT_CONFIG["lr_encoder"],
        help="Learning rate for encoder",
    )
    parser.add_argument(
        "--lr_projector",
        type=float,
        default=DEFAULT_CONFIG["lr_projector"],
        help="Learning rate for projector",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="OpenTSLM/llama-3.2-1b-m4-sp",
        help="HuggingFace model repository ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ucr_classification",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (no training)",
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


def evaluate(
    model,
    test_loader: DataLoader,
    label_set: List[str],
    constrained: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate classification accuracy on test set.
    
    Args:
        model: The OpenTSLM model
        test_loader: DataLoader for test episodes
        label_set: List of valid labels
        constrained: Whether to use constrained decoding
    
    Returns:
        Dict with accuracy and predictions
    """
    model.eval()
    
    # Create constrained logits processor
    if constrained:
        logits_processor = LabelConstrainedLogitsProcessor(
            allowed_labels=label_set,
            tokenizer=model.tokenizer,
            include_eos=True,
        )
        processor_list = [logits_processor]
    else:
        processor_list = None
    
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Generate predictions
            if processor_list:
                outputs = model.generate(
                    batch,
                    max_new_tokens=1,
                    logits_processor=processor_list,
                    do_sample=False,
                )
            else:
                outputs = model.generate(
                    batch,
                    max_new_tokens=1,
                    do_sample=False,
                )
            
            # Compare with ground truth
            for sample, pred in zip(batch, outputs):
                # Extract the predicted label (first character after any whitespace)
                pred_label = pred.strip()
                if pred_label:
                    pred_label = pred_label[0].upper()
                
                gold_label = sample["_query_label"]
                is_correct = pred_label == gold_label
                
                if is_correct:
                    correct += 1
                total += 1
                
                predictions.append({
                    "predicted": pred_label,
                    "gold": gold_label,
                    "correct": is_correct,
                    "raw_output": pred,
                })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    label_set: List[str],
    args,
    output_dir: str,
):
    """
    Train the model for classification.
    
    Args:
        model: OpenTSLM model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        label_set: List of valid labels
        args: Command line arguments
        output_dir: Directory to save results
    """
    device = model.device if hasattr(model, 'device') else 'cuda'
    
    # Freeze LLM, only train encoder and projector
    for p in model.llm.parameters():
        p.requires_grad = False
    
    # Setup optimizer with different learning rates
    enc_params = list(model.encoder.parameters())
    proj_params = list(model.projector.projector.parameters())
    
    optimizer = AdamW([
        {"params": enc_params, "lr": args.lr_encoder, "weight_decay": DEFAULT_CONFIG["weight_decay"]},
        {"params": proj_params, "lr": args.lr_projector, "weight_decay": DEFAULT_CONFIG["weight_decay"]},
    ])
    
    # Scheduler
    total_steps = args.num_epochs * len(train_loader)
    warmup_steps = int(DEFAULT_CONFIG["warmup_frac"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\n📈 Training configuration:")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Encoder LR: {args.lr_encoder:.2e}")
    print(f"   Projector LR: {args.lr_projector:.2e}")
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    training_history = []
    
    for epoch in range(1, args.num_epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch in prog:
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            prog.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")
        
        # Validation
        if epoch % DEFAULT_CONFIG["eval_every_n_epochs"] == 0:
            val_results = evaluate(model, val_loader, label_set, constrained=True)
            val_acc = val_results["accuracy"]
            print(f"Epoch {epoch} — val accuracy: {val_acc:.4f}")
            
            training_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_accuracy": val_acc,
            })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "encoder_state": model.encoder.state_dict(),
                    "projector_state": model.projector.state_dict(),
                    "val_accuracy": val_acc,
                }, checkpoint_path)
                print(f"✔️  New best model saved (val_acc: {val_acc:.4f})")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{DEFAULT_CONFIG['early_stop_patience']} epochs")
                
                if epochs_no_improve >= DEFAULT_CONFIG["early_stop_patience"]:
                    print("\nEarly stopping triggered.")
                    break
    
    # Load best model for final evaluation
    best_checkpoint = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_checkpoint):
        ckpt = torch.load(best_checkpoint, map_location=device)
        model.encoder.load_state_dict(ckpt["encoder_state"])
        model.projector.load_state_dict(ckpt["projector_state"])
        print(f"\n📂 Loaded best model from epoch {ckpt['epoch']}")
    
    # Final test evaluation
    print("\n🧪 Running final test evaluation...")
    test_results = evaluate(model, test_loader, label_set, constrained=True)
    print(f"🎯 Test Accuracy: {test_results['accuracy']:.4f} ({test_results['correct']}/{test_results['total']})")
    
    # Save all results
    results = {
        "config": vars(args),
        "training_history": training_history,
        "test_results": {
            "accuracy": test_results["accuracy"],
            "correct": test_results["correct"],
            "total": test_results["total"],
        },
        "best_val_accuracy": best_val_acc,
    }
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📊 Results saved to {results_path}")
    
    # Save predictions
    predictions_path = os.path.join(output_dir, "test_predictions.jsonl")
    with open(predictions_path, "w") as f:
        for pred in test_results["predictions"]:
            f.write(json.dumps(pred) + "\n")
    print(f"📝 Predictions saved to {predictions_path}")
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("M1: UCR Time Series Classification")
    print("Soft Prompt + Random Sampling")
    print("=" * 60)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, args.dataset_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load model from HuggingFace
    print(f"\n📥 Loading model from {args.repo_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OpenTSLM.load_pretrained(args.repo_id, device=device)
    
    # Create datasets
    print(f"\n📊 Creating datasets for {args.dataset_name}...")
    train_dataset = UCREpisodeDataset(
        dataset_name=args.dataset_name,
        split="train",
        n_support=args.n_support,
        EOS_TOKEN=model.get_eos_token(),
        seed=args.seed,
    )
    
    # For validation, use a portion of training data with different seed
    val_dataset = UCREpisodeDataset(
        dataset_name=args.dataset_name,
        split="train",  # Use train split for validation too
        n_support=args.n_support,
        EOS_TOKEN=model.get_eos_token(),
        seed=args.seed + 1,  # Different seed for different episodes
    )
    
    test_dataset = UCREpisodeDataset(
        dataset_name=args.dataset_name,
        split="test",
        n_support=args.n_support,
        EOS_TOKEN=model.get_eos_token(),
        seed=args.seed + 2,
    )
    
    # Get label set
    label_set = train_dataset.get_label_set()
    print(f"   Label set: {label_set}")
    
    # Create data loaders
    collate_fn = lambda batch: collate_ucr_episodes(batch, patch_size=PATCH_SIZE)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    if args.eval_only:
        print("\n🔍 Evaluation-only mode...")
        results = evaluate(model, test_loader, label_set, constrained=True)
        print(f"🎯 Test Accuracy: {results['accuracy']:.4f}")
    else:
        # Train
        results = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            label_set=label_set,
            args=args,
            output_dir=output_dir,
        )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
