#!/usr/bin/env python3
"""
Training script for VLA fine-tuning on RoboVQA.

Usage:
    # Basic usage
    python scripts/train.py --config configs/stage1.yaml
    
    # Override any config key with dot notation
    python scripts/train.py --config configs/stage1.yaml \
        data.subset_ratio=0.1 \
        optimizer.learning_rate=1e-4 \
        batch_size=2
    
    # Dry run (validate setup only)
    python scripts/train.py --config configs/stage1.yaml --dry_run
    
    # Estimate memory
    python scripts/train.py --config configs/stage1.yaml --estimate_memory
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlaft.training.config import TrainingConfig, load_config
from vlaft.training.trainer import VLATrainer
from vlaft.models.internvl import estimate_memory_usage

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VLA model on RoboVQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --config configs/stage1.yaml
  python scripts/train.py --config configs/stage1.yaml data.subset_ratio=0.1
  python scripts/train.py --config configs/stage1.yaml batch_size=2 optimizer.learning_rate=1e-4
        """,
    )
    
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Validate setup only (no training)"
    )
    parser.add_argument(
        "--estimate_memory", action="store_true",
        help="Estimate memory usage and exit"
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dot notation (e.g., data.subset_ratio=0.1)"
    )
    
    return parser.parse_args()


def apply_overrides(config_dict: dict, overrides: list) -> dict:
    """
    Apply dot-notation overrides to config dictionary.
    
    Examples:
        data.subset_ratio=0.1 -> config_dict["data"]["subset_ratio"] = 0.1
        batch_size=2 -> config_dict["batch_size"] = 2
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override} (expected key=value)")
        
        key, value = override.split("=", 1)
        keys = key.split(".")
        
        # Parse value type
        value = parse_value(value)
        
        # Navigate to nested key
        d = config_dict
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        
        d[keys[-1]] = value
        logger.info(f"Override: {key} = {value}")
    
    return config_dict


def parse_value(value: str):
    """Parse string value to appropriate type."""
    # None
    if value.lower() == "null" or value.lower() == "none":
        return None
    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # String
    return value


def validate_data_exists(config: TrainingConfig) -> bool:
    """Validate that required data files exist."""
    data_dir = Path(config.data.data_dir)
    
    required = [
        data_dir / "images",
        data_dir / f"stage{config.data.stage}" / "train.jsonl",
        data_dir / f"stage{config.data.stage}" / "val.jsonl",
    ]
    
    missing = [str(p) for p in required if not p.exists()]
    
    if missing:
        logger.error("Missing required data:")
        for path in missing:
            logger.error(f"  - {path}")
        return False
    
    return True


def print_config_summary(config: TrainingConfig):
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Stage:            {config.data.stage}")
    print(f"Model:            {config.model.model_name}")
    print(f"Data subset:      {config.data.subset_ratio * 100:.0f}%")
    print(f"Batch size:       {config.batch_size} × {config.gradient_accumulation_steps} = {config.effective_batch_size}")
    print(f"Epochs:           {config.num_epochs}")
    print(f"Learning rate:    {config.optimizer.learning_rate}")
    print(f"LoRA rank:        {config.model.lora_r}")
    print(f"Output:           {config.output_dir}")
    print("=" * 60 + "\n")


def print_memory_estimate(config: TrainingConfig):
    """Print memory usage estimate."""
    estimates = estimate_memory_usage(
        model_name=config.model.model_name,
        batch_size=config.batch_size,
        max_frames=config.data.max_frames,
        sequence_length=config.data.max_length,
        use_qlora=config.model.use_qlora,
        gradient_checkpointing=config.model.gradient_checkpointing,
    )
    
    print("\n" + "=" * 60)
    print("Memory Usage Estimate")
    print("=" * 60)
    print(f"Model memory:          {estimates['model_memory_gb']:.1f} GB")
    print(f"LoRA memory:           {estimates['lora_memory_gb']:.1f} GB")
    print(f"Optimizer memory:      {estimates['optimizer_memory_gb']:.1f} GB")
    print(f"Gradient memory:       {estimates['gradient_memory_gb']:.1f} GB")
    print(f"Activation memory:     {estimates['activation_memory_gb']:.1f} GB")
    print("-" * 60)
    print(f"TOTAL ESTIMATED:       {estimates['total_estimated_gb']:.1f} GB")
    print(f"DGX Spark headroom:    ~{128 - estimates['total_estimated_gb']:.1f} GB")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Load config
    logger.info(f"Loading config: {args.config}")
    config_dict = load_config(args.config).to_dict()
    
    # Apply overrides
    if args.overrides:
        config_dict = apply_overrides(config_dict, args.overrides)
    
    config = TrainingConfig.from_dict(config_dict)
    
    # Memory estimation only
    if args.estimate_memory:
        print_memory_estimate(config)
        return
    
    # Validate data
    if not validate_data_exists(config):
        logger.error("Data validation failed. Run preprocessing first.")
        sys.exit(1)
    
    print_config_summary(config)
    
    # Dry run
    if args.dry_run:
        logger.info("Dry run complete - config is valid")
        return
    
    # Train
    trainer = VLATrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()