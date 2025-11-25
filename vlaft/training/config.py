"""Training configuration for VLA fine-tuning."""

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: str = "data/robovqa/processed"
    images_subdir: str = "images"
    stage: int = 1
    max_frames: int = 16
    max_length: int = 2048
    max_dynamic_patch: int = 1  # 1 = fixed resolution (no dynamic tiling)
    subset_ratio: float = 1.0
    max_samples: Optional[int] = None
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "InternVL3-8B"
    model_path: Optional[str] = None  # Override HF path
    use_qlora: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = True

    # Memory override for unified memory systems (DGX Spark)
    # e.g., "120GiB" - set to available GPU memory
    max_memory_gb: Optional[int] = None

    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    warmup_steps: Optional[int] = None
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0

    # AdamW betas
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Experiment metadata
    experiment_name: str = "robovqa_stage1"
    output_dir: str = "experiments"
    seed: int = 42

    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_steps: Optional[int] = None  # Override num_epochs

    # Evaluation
    eval_steps: int = 500
    eval_batch_size: int = 8
    eval_samples: int = 1000  # Max eval samples

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Logging
    logging_steps: int = 10
    log_level: str = "INFO"
    wandb_enabled: bool = False
    wandb_project: str = "vla-ft"
    wandb_mode: str = "offline"

    # Nested configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self):
        """Validate configuration."""
        # Ensure nested configs are proper dataclasses
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to {path}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


def load_config(config_path: str) -> TrainingConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        TrainingConfig instance
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return TrainingConfig.from_dict(config_dict)


def get_default_stage1_config() -> TrainingConfig:
    """Get default Stage 1 configuration (split QAs)."""
    return TrainingConfig(
        experiment_name="robovqa_stage1",
        data=DataConfig(
            stage=1,
            subset_ratio=0.1,  # Start with 10%
        ),
        model=ModelConfig(
            model_name="InternVL3-8B",
            use_qlora=True,
            lora_r=64,
        ),
        optimizer=OptimizerConfig(
            learning_rate=2e-4,
            warmup_ratio=0.03,
        ),
        batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size 32
        num_epochs=3,
        eval_steps=500,
        save_steps=1000,
    )


def get_default_stage2_config() -> TrainingConfig:
    """Get default Stage 2 configuration (multi-turn)."""
    return TrainingConfig(
        experiment_name="robovqa_stage2",
        data=DataConfig(
            stage=2,
            subset_ratio=0.1,  # Start with 10%
        ),
        model=ModelConfig(
            model_name="InternVL3-8B",
            use_qlora=True,
            lora_r=64,
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-4,  # Lower LR for stage 2
            warmup_ratio=0.03,
        ),
        batch_size=2,  # Smaller batch for longer sequences
        gradient_accumulation_steps=16,  # Effective batch size 32
        num_epochs=2,
        eval_steps=250,
        save_steps=500,
    )


# Default configs as YAML strings for reference
DEFAULT_STAGE1_YAML = """
# Stage 1: Split QA Training
# Basic visual grounding with single QA pairs

experiment_name: robovqa_stage1
output_dir: experiments
seed: 42

# Training parameters
batch_size: 4
gradient_accumulation_steps: 8  # Effective batch size: 32
num_epochs: 3

# Evaluation and checkpointing
eval_steps: 500
eval_batch_size: 8
eval_samples: 1000
save_steps: 1000
save_total_limit: 3

# Logging
logging_steps: 10
wandb_enabled: false
wandb_project: vla-ft
wandb_mode: offline

# Data configuration
data:
  data_dir: data/robovqa/processed
  stage: 1
  max_frames: 16
  max_length: 2048
  subset_ratio: 0.1  # Start with 10% for validation
  num_workers: 4

# Model configuration
model:
  model_name: InternVL3-8B
  use_qlora: true
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
  gradient_checkpointing: true
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05

# Optimizer configuration
optimizer:
  optimizer: adamw
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler: cosine
  max_grad_norm: 1.0
"""

DEFAULT_STAGE2_YAML = """
# Stage 2: Multi-turn Training
# Reasoning chains with conversation context

experiment_name: robovqa_stage2
output_dir: experiments
seed: 42

# Training parameters
batch_size: 2  # Smaller for longer sequences
gradient_accumulation_steps: 16  # Effective batch size: 32
num_epochs: 2

# Evaluation and checkpointing
eval_steps: 250
eval_batch_size: 4
eval_samples: 500
save_steps: 500
save_total_limit: 3

# Logging
logging_steps: 10
wandb_enabled: false
wandb_project: vla-ft
wandb_mode: offline

# Data configuration
data:
  data_dir: data/robovqa/processed
  stage: 2
  max_frames: 16
  max_length: 4096  # Longer for multi-turn
  subset_ratio: 0.1
  num_workers: 4

# Model configuration  
model:
  model_name: InternVL3-8B
  model_path: null  # Will load from stage 1 checkpoint
  use_qlora: true
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
  gradient_checkpointing: true
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05

# Optimizer configuration
optimizer:
  optimizer: adamw
  learning_rate: 1.0e-4  # Lower LR for stage 2
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler: cosine
  max_grad_norm: 1.0
"""
