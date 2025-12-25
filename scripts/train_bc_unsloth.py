#!/usr/bin/env python3
"""
Behavioral Cloning (BC) training script using Unsloth.

Trains a LoRA adapter on Qwen for browser automation using BC dataset.

Usage:
    python scripts/train_bc_unsloth.py --dataset_path data/datasets/bc.jsonl
    python scripts/train_bc_unsloth.py --base_model Qwen/Qwen2.5-3B-Instruct --epochs 3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

load_dotenv()


def load_bc_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load BC dataset from JSONL file.
    
    Expected format per line:
    {"prompt": "...", "action": "...", "task_id": "...", "step": N}
    
    Args:
        dataset_path: Path to bc.jsonl
        
    Returns:
        List of training examples
    """
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_for_training(examples: List[Dict]) -> List[Dict]:
    """Format BC examples for Unsloth training.
    
    Converts to instruction-following format:
    - instruction: The observation/prompt
    - output: The action to take
    """
    formatted = []
    for ex in examples:
        formatted.append({
            "instruction": ex["prompt"],
            "output": json.dumps({"action": ex["action"], "rationale": "BC training example"}),
        })
    return formatted


def train_bc(
    base_model: str,
    dataset_path: str,
    output_dir: str,
    max_seq_len: int = 2048,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    warmup_ratio: float = 0.03,
    save_steps: int = 100,
    logging_steps: int = 10,
):
    """Train BC model using Unsloth.
    
    Args:
        base_model: HuggingFace model name
        dataset_path: Path to BC dataset
        output_dir: Directory to save adapter
        max_seq_len: Maximum sequence length
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        epochs: Number of training epochs
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        warmup_ratio: Warmup ratio
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
    """
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: Missing required packages: {e}")
        print("Install with: pip install unsloth trl datasets")
        sys.exit(1)
    
    print(f"{'='*60}")
    print("HALO-Agent BC Training with Unsloth")
    print(f"{'='*60}")
    print(f"Base Model: {base_model}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Max Seq Len: {max_seq_len}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"LoRA r: {lora_r}, alpha: {lora_alpha}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    raw_examples = load_bc_dataset(dataset_path)
    print(f"Loaded {len(raw_examples)} examples")
    
    formatted_examples = format_for_training(raw_examples)
    dataset = Dataset.from_list(formatted_examples)
    print(f"Dataset ready: {len(dataset)} examples")
    
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
    
    # Training arguments
    training_args = TrainingArguments(
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
        report_to="none",
    )
    
    # Format prompt function
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for inst, out in zip(instructions, outputs):
            text = f"<|im_start|>system\nYou are a browser automation agent.<|im_end|>\n<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Create trainer
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    # Save adapter
    print(f"\nSaving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'='*60}")
    print("BC Training Complete!")
    print(f"Adapter saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train BC model using Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with defaults
    python train_bc_unsloth.py --dataset_path data/datasets/bc.jsonl
    
    # Custom model and epochs
    python train_bc_unsloth.py --base_model Qwen/Qwen2.5-7B-Instruct --epochs 5
    
    # Adjust batch size for smaller GPUs
    python train_bc_unsloth.py --batch_size 2 --gradient_accumulation_steps 8
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
        default="data/datasets/bc.jsonl",
        help="Path to BC dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/qwen_bc_lora",
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
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
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
    
    train_bc(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
