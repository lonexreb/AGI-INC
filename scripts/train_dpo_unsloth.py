#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) training script using Unsloth.

Trains a LoRA adapter on Qwen using preference pairs (chosen vs rejected actions).

Usage:
    python scripts/train_dpo_unsloth.py --dataset_path data/datasets/dpo.jsonl
    python scripts/train_dpo_unsloth.py --base_model Qwen/Qwen2.5-3B-Instruct --epochs 2
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

load_dotenv()


def load_dpo_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load DPO dataset from JSONL file.
    
    Expected format per line:
    {"prompt": "...", "chosen": "...", "rejected": "...", "task_id": "...", "step": N}
    
    Args:
        dataset_path: Path to dpo.jsonl
        
    Returns:
        List of preference pairs
    """
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_for_dpo(examples: List[Dict]) -> List[Dict]:
    """Format DPO examples for training.
    
    Converts to DPO format:
    - prompt: The observation/context
    - chosen: The preferred action (successful)
    - rejected: The non-preferred action (failed)
    """
    formatted = []
    for ex in examples:
        formatted.append({
            "prompt": ex["prompt"],
            "chosen": json.dumps({"action": ex["chosen"], "rationale": "Successful action"}),
            "rejected": json.dumps({"action": ex["rejected"], "rationale": "Failed action"}),
        })
    return formatted


def train_dpo(
    base_model: str,
    dataset_path: str,
    output_dir: str,
    max_seq_len: int = 2048,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    epochs: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    beta: float = 0.1,
    warmup_ratio: float = 0.1,
    save_steps: int = 50,
    logging_steps: int = 10,
):
    """Train DPO model using Unsloth.
    
    Args:
        base_model: HuggingFace model name
        dataset_path: Path to DPO dataset
        output_dir: Directory to save adapter
        max_seq_len: Maximum sequence length
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        epochs: Number of training epochs
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        beta: DPO beta parameter (controls deviation from reference)
        warmup_ratio: Warmup ratio
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
    """
    try:
        from unsloth import FastLanguageModel
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: Missing required packages: {e}")
        print("Install with: pip install unsloth trl datasets")
        sys.exit(1)
    
    print(f"{'='*60}")
    print("HALO-Agent DPO Training with Unsloth")
    print(f"{'='*60}")
    print(f"Base Model: {base_model}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Max Seq Len: {max_seq_len}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"DPO Beta: {beta}")
    print(f"LoRA r: {lora_r}, alpha: {lora_alpha}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    raw_examples = load_dpo_dataset(dataset_path)
    print(f"Loaded {len(raw_examples)} preference pairs")
    
    formatted_examples = format_for_dpo(raw_examples)
    dataset = Dataset.from_list(formatted_examples)
    print(f"Dataset ready: {len(dataset)} pairs")
    
    # Load model with Unsloth
    print("\nLoading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # DPO config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=42,
        beta=beta,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create DPO trainer
    print("\nStarting DPO training...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth handles this
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save adapter
    print(f"\nSaving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'='*60}")
    print("DPO Training Complete!")
    print(f"Adapter saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train DPO model using Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with defaults
    python train_dpo_unsloth.py --dataset_path data/datasets/dpo.jsonl
    
    # Custom model and epochs
    python train_dpo_unsloth.py --base_model Qwen/Qwen2.5-7B-Instruct --epochs 3
    
    # Adjust beta for stronger preference learning
    python train_dpo_unsloth.py --beta 0.2
"""
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/datasets/dpo.jsonl",
        help="Path to DPO dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/qwen_dpo_lora",
        help="Output directory for adapter"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    
    args = parser.parse_args()
    
    # Check dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"ERROR: Dataset not found: {args.dataset_path}")
        print("Run 'python scripts/collect_traj.py' to generate the dataset first.")
        sys.exit(1)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train_dpo(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        epochs=args.epochs,
        beta=args.beta,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
