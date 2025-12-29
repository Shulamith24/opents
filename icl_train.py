#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
M1 Ablation Experiment: ICL Time Series Classification Training Script.

This script trains OpenTSLM on UCR datasets using an In-Context Learning (ICL)
approach with episode-based sampling and LoRA fine-tuning.

Key Features:
- Loads pretrained model from HuggingFace (stage2 M4 weights)
- Enables LoRA for efficient fine-tuning
- Uses UCRICLDataset for episode-based sampling
- Supports constrained decoding for evaluation

Usage:
    python icl_train.py --dataset ECG5000 --k_shot 1 --epochs 30

"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from model.llm.OpenTSLM import OpenTSLM
from time_series_datasets.ucr.UCRICLDataset import UCRICLDataset
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from constrained_decoding import LabelConstrainedLogitsProcessor
from model_config import PATCH_SIZE


# Default hyperparameters (aligned with curriculum_learning.py stage3)
DEFAULT_CONFIG = {
    "batch_size": 4,
    "epochs": 30,
    "lr_encoder": 2e-4,
    "lr_projector": 1e-4,
    "lora_lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_frac": 0.1,
    "grad_clip_norm": 1.0,
    "early_stop_patience": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(description="M1 Ablation: ICL Time Series Classification")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="ECG5000", help="UCR dataset name (e.g., ECG5000)")
    parser.add_argument("--k_shot", type=int, default=3,help="Number of support examples per class")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to UCR data directory")
    
    # Model arguments
    parser.add_argument("--repo_id", type=str, default="OpenTSLM/llama-3.2-1b-m4-sp", help="HuggingFace repository ID for pretrained model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr_encoder", type=float, default=DEFAULT_CONFIG["lr_encoder"])
    parser.add_argument("--lr_projector", type=float, default=DEFAULT_CONFIG["lr_projector"])
    parser.add_argument("--lora_lr", type=float, default=DEFAULT_CONFIG["lora_lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--warmup_frac", type=float, default=DEFAULT_CONFIG["warmup_frac"])
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_CONFIG["grad_clip_norm"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["early_stop_patience"])
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results/icl_m1",help="Output directory for checkpoints and results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true",help="Only run evaluation (requires checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None,help="Path to checkpoint for evaluation")
    
    # Encoder selection (M2: TSLANet)
    parser.add_argument(
        "--encoder", type=str, default="transformer_cnn",
        choices=["transformer_cnn", "tslanet"],
        help="Encoder type to use"
    )
    parser.add_argument(
        "--tslanet_checkpoint", type=str, default=None,
        help="Path to TSLANet pretrained checkpoint"
    )
    parser.add_argument("--tslanet_depth", type=int, default=2, help="TSLANet depth")
    parser.add_argument("--tslanet_patch_size", type=int, default=8, help="TSLANet patch size")
    
    # RAG arguments (M3)
    parser.add_argument(
        "--rag_mode", type=str, default="none",
        choices=["none", "eval", "train"],
        help="RAG mode: none=random support, eval=RAG for eval only, train=RAG for both"
    )
    parser.add_argument(
        "--rag_index_path", type=str, default=None,
        help="Path to prebuilt RAG index (required if rag_mode != none)"
    )
    parser.add_argument(
        "--rag_method", type=str, default="faiss",
        choices=["faiss", "brute"],
        help="RAG index method"
    )
    parser.add_argument(
        "--rag_top_m", type=int, default=50,
        help="Top-M candidates for RAG retrieval"
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
        f"k{args.k_shot}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    return output_dir


def get_optimizer(model, args):
    """Create optimizer with parameter groups."""
    # Get underlying model if wrapped
    base_model = model.module if hasattr(model, "module") else model
    
    # Parameter groups
    param_groups = [
        {
            "params": base_model.encoder.parameters(),
            "lr": args.lr_encoder,
            "weight_decay": args.weight_decay,
        },
        {
            "params": base_model.projector.parameters(),
            "lr": args.lr_projector,
            "weight_decay": args.weight_decay,
        },
    ]
    
    # Add LoRA parameters if enabled
    if hasattr(base_model, "lora_enabled") and base_model.lora_enabled:
        lora_params = base_model.get_lora_parameters()
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": args.lora_lr,
                "weight_decay": args.weight_decay,
            })
    
    # Print training configuration
    print(f"\n📊 Training Configuration:")
    print(f"   ✅ Encoder: lr={args.lr_encoder:.2e} (trainable)")
    print(f"   ✅ Projector: lr={args.lr_projector:.2e} (trainable)")
    if hasattr(base_model, "lora_enabled") and base_model.lora_enabled:
        lora_params = base_model.get_lora_parameters()
        print(f"   ✅ LLM (LoRA): lr={args.lora_lr:.2e} ({len(lora_params)} params)")
        print(f"   ❄️  LLM (base): frozen")
    else:
        print(f"   ❄️  LLM: frozen (no LoRA)")
    
    return AdamW(param_groups)


def collate_fn(batch):
    """Collate function for DataLoader."""
    return extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=PATCH_SIZE
    )


def train_epoch(model, train_loader, optimizer, scheduler, args, epoch) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    # Get underlying model
    base_model = model.module if hasattr(model, "module") else model
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        optimizer.zero_grad()
        
        loss = base_model.compute_loss(batch)
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(base_model.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    
    return total_loss / len(train_loader)


def validate(model, val_loader, args) -> float:
    """Validate and return average loss."""
    model.eval()
    total_loss = 0.0
    
    # Get underlying model
    base_model = model.module if hasattr(model, "module") else model
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            loss = base_model.compute_loss(batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def evaluate(
    model,
    test_loader,
    num_classes: int,
    output_dir: str,
    use_constrained_decoding: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model with accuracy metric.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test set
        num_classes: Number of classes for constrained decoding
        output_dir: Directory to save results
        use_constrained_decoding: Whether to use constrained decoding
    
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    # Get underlying model
    base_model = model.module if hasattr(model, "module") else model
    
    # Create constrained decoder if needed
    logits_processor = None
    if use_constrained_decoding:
        valid_labels = [chr(ord('A') + i) for i in range(num_classes)]
        logits_processor = LabelConstrainedLogitsProcessor(
            base_model.tokenizer,
            valid_labels
        )
    
    # Collect predictions
    results = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Generate with optional constrained decoding
            generate_kwargs = {"max_new_tokens": 5}
            if logits_processor:
                generate_kwargs["logits_processor"] = [logits_processor]
            
            predictions = base_model.generate(batch, **generate_kwargs)
            
            for sample, pred in zip(batch, predictions):
                # Extract predicted label (first non-whitespace character)
                pred_clean = pred.strip()
                if pred_clean:
                    pred_label = pred_clean[0].upper()
                else:
                    pred_label = "?"
                
                gold_label = sample.get("query_label_str", sample.get("answer", "?")[0])
                
                is_correct = pred_label == gold_label
                if is_correct:
                    correct += 1
                total += 1
                
                result = {
                    "prediction": pred_clean,
                    "predicted_label": pred_label,
                    "gold_label": gold_label,
                    "correct": is_correct,
                }
                results.append(result)
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
    
    # Save results
    results_file = os.path.join(output_dir, "results", "test_predictions.jsonl")
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    metrics_file = os.path.join(output_dir, "results", "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"   Results saved to: {results_file}")
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, output_dir):
    """Save model checkpoint."""
    # Get underlying model
    base_model = model.module if hasattr(model, "module") else model
    
    checkpoint = {
        "encoder_state": base_model.encoder.state_dict(),
        "projector_state": base_model.projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    
    # Save LoRA state
    base_model.save_lora_state_to_checkpoint(checkpoint)
    
    checkpoint_path = os.path.join(output_dir, "checkpoints", "best_model.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    # Get underlying model
    base_model = model.module if hasattr(model, "module") else model
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    base_model.encoder.load_state_dict(checkpoint["encoder_state"])
    base_model.projector.load_state_dict(checkpoint["projector_state"])
    
    # Load LoRA state
    base_model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    if scheduler and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    
    print(f"📂 Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    return checkpoint.get("epoch", 0), checkpoint.get("val_loss", float("inf"))


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 60)
    print("M1 Ablation: ICL Time Series Classification")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"K-shot: {args.k_shot}")
    print(f"Model: {args.repo_id}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_output_dir(args)
    print(f"📁 Output directory: {output_dir}")
    
    # Load pretrained model from HuggingFace
    print(f"\n📥 Loading pretrained model from {args.repo_id}...")
    model = OpenTSLM.load_pretrained(
        args.repo_id,
        device=args.device,
        enable_lora=True,  # Enable LoRA for fine-tuning
    )
    
    # Re-enable LoRA with custom config if needed
    base_model = model.module if hasattr(model, "module") else model
    if not base_model.lora_enabled:
        print("🔧 Enabling LoRA...")
        base_model.enable_lora(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    
    # Replace encoder if using TSLANet (M2)
    if args.encoder == "tslanet":
        print(f"\n🔄 Replacing encoder with TSLANet...")
        from model.encoder.TSLANetEncoder import TSLANetEncoder
        
        # Determine max_seq_len from checkpoint if provided
        max_seq_len = 4096  # default
        if args.tslanet_checkpoint:
            print(f"   Loading pretrained weights from {args.tslanet_checkpoint}")
            state_dict = torch.load(args.tslanet_checkpoint, map_location="cpu", weights_only=True)
            # Handle both full checkpoint and state_dict only
            if "encoder_state" in state_dict:
                encoder_state = state_dict["encoder_state"]
            else:
                encoder_state = state_dict
            
            # Infer max_seq_len from pos_embed shape
            if "pos_embed" in encoder_state:
                num_patches = encoder_state["pos_embed"].shape[1]
                # Reverse calculate max_seq_len from num_patches
                # num_patches = (seq_len - patch_size) / stride + 1, stride = patch_size // 2
                stride = args.tslanet_patch_size // 2
                max_seq_len = (num_patches - 1) * stride + args.tslanet_patch_size
                print(f"   Inferred max_seq_len={max_seq_len} from checkpoint (num_patches={num_patches})")
        
        new_encoder = TSLANetEncoder(
            output_dim=base_model.encoder.output_dim,
            patch_size=args.tslanet_patch_size,
            depth=args.tslanet_depth,
            dropout=0.15,
            max_seq_len=max_seq_len,
        )
        
        if args.tslanet_checkpoint:
            new_encoder.load_state_dict(encoder_state)
        
        base_model.encoder = new_encoder.to(args.device)
        print(f"   ✅ TSLANet encoder loaded (depth={args.tslanet_depth}, patch_size={args.tslanet_patch_size})")
    
    # Load RAG index if needed
    rag_index = None
    if args.rag_mode != "none":
        if not args.rag_index_path:
            raise ValueError("--rag_index_path required when rag_mode != none")
        
        print(f"\n📚 Loading RAG index from {args.rag_index_path}...")
        from rag.rag_index import RAGIndex
        rag_index = RAGIndex(method=args.rag_method)
        rag_index.load(args.rag_index_path)
    
    # Create datasets
    print(f"\n📊 Creating datasets...")
    print(f"   RAG mode: {args.rag_mode}")
    
    if args.rag_mode == "train":
        # Both train and test use RAG
        from time_series_datasets.ucr.UCRRAGDataset import UCRRAGDataset
        train_dataset = UCRRAGDataset(
            rag_index=rag_index,
            encoder=base_model.encoder,
            device=args.device,
            top_m=args.rag_top_m,
            dataset_name=args.dataset,
            split="train",
            k_shot=args.k_shot,
            EOS_TOKEN=base_model.get_eos_token(),
            raw_data_path=args.data_path,
            seed=args.seed,
        )
        test_dataset = UCRRAGDataset(
            rag_index=rag_index,
            encoder=base_model.encoder,
            device=args.device,
            top_m=args.rag_top_m,
            dataset_name=args.dataset,
            split="test",
            k_shot=args.k_shot,
            EOS_TOKEN=base_model.get_eos_token(),
            raw_data_path=args.data_path,
            seed=args.seed + 1,
        )
    elif args.rag_mode == "eval":
        # Train uses random, test uses RAG
        train_dataset = UCRICLDataset(
            dataset_name=args.dataset,
            split="train",
            k_shot=args.k_shot,
            EOS_TOKEN=base_model.get_eos_token(),
            raw_data_path=args.data_path,
            seed=args.seed,
        )
        from time_series_datasets.ucr.UCRRAGDataset import UCRRAGDataset
        test_dataset = UCRRAGDataset(
            rag_index=rag_index,
            encoder=base_model.encoder,
            device=args.device,
            top_m=args.rag_top_m,
            dataset_name=args.dataset,
            split="test",
            k_shot=args.k_shot,
            EOS_TOKEN=base_model.get_eos_token(),
            raw_data_path=args.data_path,
            seed=args.seed + 1,
        )
    else:
        # Both use random (original behavior)
        train_dataset = UCRICLDataset(
            dataset_name=args.dataset,
            split="train",
            k_shot=args.k_shot,
            EOS_TOKEN=base_model.get_eos_token(),
            raw_data_path=args.data_path,
            seed=args.seed,
        )
        test_dataset = UCRICLDataset(
            dataset_name=args.dataset,
            split="test",
            k_shot=args.k_shot,
            EOS_TOKEN=base_model.get_eos_token(),
            raw_data_path=args.data_path,
            seed=args.seed + 1,
        )
    
    num_classes = train_dataset.num_classes
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Classes: {num_classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Batch size 1 for evaluation
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Eval-only mode
    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for eval_only mode")
        load_checkpoint(model, args.checkpoint)
        evaluate(model, test_loader, num_classes, output_dir)
        return
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, args)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, args, epoch)
        print(f"Epoch {epoch} — Train loss: {train_loss:.4f}")
        
        # Validate (use test set as validation for simplicity)
        val_loss = validate(model, test_loader, args)
        print(f"Epoch {epoch} — Val loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, output_dir)
            print("✔️ New best model saved!")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{args.patience} epochs")
            
            if epochs_no_improve >= args.patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break
    
    # Load best model and evaluate
    best_checkpoint = os.path.join(output_dir, "checkpoints", "best_model.pt")
    if os.path.exists(best_checkpoint):
        load_checkpoint(model, best_checkpoint)
    
    print("\n🎯 Final Evaluation...")
    metrics = evaluate(model, test_loader, num_classes, output_dir)
    
    print("\n" + "=" * 60)
    print("🏁 Training Complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Test accuracy: {metrics['accuracy']:.4f}")
    print(f"   Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
