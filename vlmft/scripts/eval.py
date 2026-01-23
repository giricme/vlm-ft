#!/usr/bin/env python3
"""
Evaluate a checkpoint on Stage 1 or Stage 2 validation data.

Usage:
    python -m vlmft.scripts.eval \
        --checkpoint experiments/robovqa_stage2_only_xxx/checkpoints/final \
        --stage 1 \
        --max_samples 500

    # Or with config overrides:
    python -m vlmft.scripts.eval \
        --checkpoint experiments/robovqa_stage2_xxx/checkpoints/best \
        --stage 2 \
        --data.subset_ratio 1.0
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# Setup logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on RoboVQA data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (contains adapter/ and training_state.pt)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2],
        help="Evaluation stage (1 or 2)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/robovqa/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=10,
        help="Maximum frames per sample",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=3,
        help="Number of example predictions to show (0 to disable)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """Load model with LoRA weights from checkpoint."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from vlmft.models.internvl import load_internvl3

    logger.info("Loading base model...")
    model, tokenizer = load_internvl3(
        model_name="InternVL3-8B",
        use_qlora=True,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        gradient_checkpointing=False,  # Not needed for eval
        max_memory={0: "115GiB"},
        max_dynamic_patch=1,
        lora_config={
            "lora_r": 128,
            "lora_alpha": 256,
            "lora_dropout": 0.0,  # No dropout for eval
            "target_modules": None,
        },
    )

    # Load LoRA weights
    checkpoint_dir = Path(checkpoint_path)
    adapter_path = checkpoint_dir / "adapter"

    if adapter_path.exists():
        adapter_weights_path = adapter_path / "adapter_model.safetensors"
        if adapter_weights_path.exists():
            logger.info(f"Loading adapter weights from {adapter_weights_path}")
            adapter_state_dict = load_file(str(adapter_weights_path))
        else:
            adapter_weights_path = adapter_path / "adapter_model.bin"
            if adapter_weights_path.exists():
                logger.info(f"Loading adapter weights from {adapter_weights_path}")
                adapter_state_dict = torch.load(adapter_weights_path, map_location=device)
            else:
                raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

        set_peft_model_state_dict(model, adapter_state_dict)
        logger.info("Adapter weights loaded successfully")
    else:
        logger.warning(f"No adapter directory found at {adapter_path}, using base model")

    model.eval()
    return model, tokenizer


def create_eval_dataloader(
    data_dir: str,
    stage: int,
    tokenizer,
    model,
    batch_size: int,
    max_samples: int,
    max_frames: int,
    max_length: int,
):
    """Create evaluation dataloader for specified stage."""
    from vlmft.data.dataloader import load_stage_data

    logger.info(f"Loading Stage {stage} validation data...")
    dataloader = load_stage_data(
        data_dir=data_dir,
        stage=stage,
        split="val",
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_samples=max_samples,
        num_workers=4,
        shuffle=False,
        max_frames=max_frames,
        max_length=max_length,
        max_dynamic_patch=1,
    )
    logger.info(f"Loaded {len(dataloader.dataset)} samples")
    return dataloader


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device, show_examples: int = 0):
    """Run evaluation and return metrics."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_losses = []

    dtype = torch.bfloat16

    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            outputs = model.base_model.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                image_flags=batch["image_flags"],
                labels=batch["labels"],
            )

        batch_size = batch["input_ids"].shape[0]
        batch_loss = outputs.loss.item()
        
        total_loss += batch_loss * batch_size
        total_samples += batch_size
        all_losses.append(batch_loss)

    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "samples": total_samples,
        "min_batch_loss": min(all_losses),
        "max_batch_loss": max(all_losses),
    }

    return metrics


@torch.no_grad()
def generate_examples(model, tokenizer, dataloader, device: torch.device, num_examples: int = 3):
    """Generate example predictions for qualitative analysis."""
    model.eval()
    examples = []
    
    dtype = torch.bfloat16
    dataset = dataloader.dataset

    for i in range(min(num_examples, len(dataset))):
        sample = dataset[i]
        
        # Get raw question/answer before collation
        question = sample["question"]
        ground_truth = sample["answer"]
        
        # For generation, we need to process single sample
        # This is a simplified version - full generation would need proper handling
        examples.append({
            "id": sample["metadata"]["id"],
            "task_type": sample["metadata"]["task_type"],
            "question": question[:200] + "..." if len(question) > 200 else question,
            "ground_truth": ground_truth,
        })

    return examples


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 60)
    logger.info("RoboVQA Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    # Create dataloader
    dataloader = create_eval_dataloader(
        data_dir=args.data_dir,
        stage=args.stage,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_frames=args.max_frames,
        max_length=args.max_length,
    )

    # Run evaluation
    logger.info("Running evaluation...")
    metrics = evaluate(model, dataloader, device, args.show_examples)

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Stage {args.stage} Evaluation:")
    logger.info(f"  Loss:       {metrics['loss']:.4f}")
    logger.info(f"  Perplexity: {metrics['perplexity']:.4f}")
    logger.info(f"  Samples:    {metrics['samples']}")
    logger.info(f"  Batch loss range: [{metrics['min_batch_loss']:.4f}, {metrics['max_batch_loss']:.4f}]")
    logger.info("=" * 60)

    # Show examples
    if args.show_examples > 0:
        logger.info("\nExample samples:")
        examples = generate_examples(model, tokenizer, dataloader, device, args.show_examples)
        for i, ex in enumerate(examples):
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(f"ID: {ex['id']}")
            logger.info(f"Task: {ex['task_type']}")
            logger.info(f"Q: {ex['question']}")
            logger.info(f"A: {ex['ground_truth']}")

    # Save results
    if args.output:
        results = {
            "checkpoint": args.checkpoint,
            "stage": args.stage,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "config": {
                "max_samples": args.max_samples,
                "batch_size": args.batch_size,
                "max_frames": args.max_frames,
                "max_length": args.max_length,
            },
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return metrics


if __name__ == "__main__":
    main()