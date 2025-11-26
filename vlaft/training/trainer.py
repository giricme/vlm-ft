"""Trainer for VLA fine-tuning."""

from datetime import datetime
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, Optional

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from vlaft.data.dataloader import load_stage_data
from vlaft.models.internvl import load_internvl3, save_lora_weights
from vlaft.training.config import TrainingConfig
from vlaft.common.logging_utils import CSVLogger

logger = logging.getLogger(__name__)


class VLATrainer:
    """
    Trainer for VLA fine-tuning on RoboVQA.

    Supports:
    - QLoRA training with gradient checkpointing
    - Mixed precision training (bfloat16)
    - Gradient accumulation
    - Checkpoint saving/resuming
    - WandB logging (optional)
    """

    def __init__(self, config: TrainingConfig, exp_dir: Path, csv_logger: CSVLogger):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            exp_dir: Experiment directory (created by entry point)
            csv_logger: CSV logger instance (created by entry point)
        """
        self.config = config
        self.exp_dir = exp_dir
        self.csv_logger = csv_logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seed
        self._set_seed(config.seed)

        # State tracking
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Will be initialized in setup()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.scaler = None
        self.wandb_logger = None

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logger.info(f"Set random seed: {seed}")

    def setup(self):
        """
        Initialize model, data, optimizer, and scheduler.

        Call this before training.
        """
        logger.info("Setting up trainer...")

        # Load model
        logger.info(f"Loading model: {self.config.model.model_name}")

        # Build max_memory dict if specified
        max_memory = None
        if self.config.model.max_memory_gb:
            max_memory = {0: f"{self.config.model.max_memory_gb}GiB"}
            logger.info(f"Using max_memory override: {max_memory}")

        self.model, self.tokenizer = load_internvl3(
            model_name=self.config.model.model_name,
            model_path=self.config.model.model_path,
            use_qlora=self.config.model.use_qlora,
            torch_dtype=self.config.model.torch_dtype,
            attn_implementation=self.config.model.attn_implementation,
            gradient_checkpointing=self.config.model.gradient_checkpointing,
            max_memory=max_memory,
            max_dynamic_patch=self.config.data.max_dynamic_patch,
            lora_config={
                "lora_r": self.config.model.lora_r,
                "lora_alpha": self.config.model.lora_alpha,
                "lora_dropout": self.config.model.lora_dropout,
                "target_modules": self.config.model.lora_target_modules,
            },
        )

        # Load data
        logger.info(f"Loading Stage {self.config.data.stage} data...")
        self.train_dataloader = load_stage_data(
            data_dir=self.config.data.data_dir,
            stage=self.config.data.stage,
            split="train",
            tokenizer=self.tokenizer,
            model=self.model,
            batch_size=self.config.batch_size,
            subset_ratio=self.config.data.subset_ratio,
            max_samples=self.config.data.max_samples,
            num_workers=self.config.data.num_workers,
            shuffle=True,
            max_frames=self.config.data.max_frames,
            max_length=self.config.data.max_length,
            max_dynamic_patch=self.config.data.max_dynamic_patch,
        )

        self.eval_dataloader = load_stage_data(
            data_dir=self.config.data.data_dir,
            stage=self.config.data.stage,
            split="val",
            tokenizer=self.tokenizer,
            model=self.model,
            batch_size=self.config.eval_batch_size,
            max_samples=self.config.eval_samples,
            num_workers=self.config.data.num_workers,
            shuffle=False,
            max_frames=self.config.data.max_frames,
            max_length=self.config.data.max_length,
            max_dynamic_patch=self.config.data.max_dynamic_patch,
        )

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler (for non-bfloat16)
        if self.config.model.torch_dtype != "bfloat16":
            self.scaler = GradScaler()

        # Optional WandB
        if self.config.wandb_enabled:
            self._setup_wandb()

        logger.info("Trainer setup complete!")
        self._log_training_info()

    def _create_optimizer(self) -> AdamW:
        """Create optimizer for trainable parameters."""
        opt_config = self.config.optimizer

        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = AdamW(
            trainable_params,
            lr=opt_config.learning_rate,
            betas=(opt_config.adam_beta1, opt_config.adam_beta2),
            eps=opt_config.adam_epsilon,
            weight_decay=opt_config.weight_decay,
        )

        logger.info(
            f"Created AdamW optimizer: lr={opt_config.learning_rate}, "
            f"weight_decay={opt_config.weight_decay}"
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        opt_config = self.config.optimizer

        # Calculate total steps
        steps_per_epoch = (
            len(self.train_dataloader) // self.config.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self.config.num_epochs

        if self.config.max_steps is not None:
            total_steps = min(total_steps, self.config.max_steps)

        # Warmup steps
        if opt_config.warmup_steps is not None:
            warmup_steps = opt_config.warmup_steps
        else:
            warmup_steps = int(total_steps * opt_config.warmup_ratio)

        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Create main scheduler
        if opt_config.lr_scheduler == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=opt_config.learning_rate * 0.1,
            )
        else:
            # Linear decay
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps - warmup_steps,
            )

        # Combine schedulers
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        logger.info(
            f"Created {opt_config.lr_scheduler} scheduler: "
            f"warmup={warmup_steps}, total={total_steps}"
        )

        return scheduler

    def _setup_wandb(self):
        """Initialize WandB logging."""
        try:
            import wandb

            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.to_dict(),
                mode=self.config.wandb_mode,
            )
            self.wandb_logger = wandb
            logger.info(f"WandB initialized: {self.config.wandb_project}")

        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_logger = None

    def _log_training_info(self):
        """Log training configuration summary."""
        steps_per_epoch = (
            len(self.train_dataloader) // self.config.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self.config.num_epochs

        logger.info("=" * 60)
        logger.info("Training Configuration")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model.model_name}")
        logger.info(f"QLoRA: {self.config.model.use_qlora}")
        logger.info(f"LoRA rank: {self.config.model.lora_r}")
        logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        logger.info(f"Eval samples: {len(self.eval_dataloader.dataset)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.effective_batch_size}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Learning rate: {self.config.optimizer.learning_rate}")
        logger.info("=" * 60)

    def train(self):
        """
        Main training loop.

        Runs for config.num_epochs or config.max_steps.
        """
        if self.model is None:
            raise RuntimeError("Call setup() before train()")

        logger.info("Starting training...")
        self.model.train()

        steps_per_epoch = (
            len(self.train_dataloader) // self.config.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self.config.num_epochs

        if self.config.max_steps is not None:
            total_steps = min(total_steps, self.config.max_steps)

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)

        # Training loop
        accumulated_loss = 0.0
        step_in_accumulation = 0

        progress_bar = tqdm(
            total=total_steps, desc="Training", initial=self.global_step
        )

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Timing accumulators for averaging over accumulation steps
            timing_accum = {
                "t_data": 0.0,
                "t_transfer": 0.0,
                "t_forward": 0.0,
                "t_backward": 0.0,
            }
            step_start_time = time.perf_counter()

            # Manual iteration to measure data loading time
            data_iter = iter(self.train_dataloader)
            batch_idx = 0
            
            while True:
                # Time data loading
                t_data_start = time.perf_counter()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                t_data_end = time.perf_counter()
                timing_accum["t_data"] += (t_data_end - t_data_start)
                
                # Move batch to device
                t0 = time.perf_counter()
                batch = self._to_device(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                timing_accum["t_transfer"] += (t1 - t0)

                # Forward pass with mixed precision
                loss = self._training_step(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                timing_accum["t_forward"] += (t2 - t1)

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                timing_accum["t_backward"] += (t3 - t2)

                accumulated_loss += loss.item()
                step_in_accumulation += 1
                batch_idx += 1

                # Gradient accumulation complete
                if step_in_accumulation >= self.config.gradient_accumulation_steps:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optimizer.max_grad_norm,
                    )

                    # Optimizer step
                    t_opt_start = time.perf_counter()
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_opt_end = time.perf_counter()
                    t_optimizer = t_opt_end - t_opt_start

                    # Total step time
                    t_step = time.perf_counter() - step_start_time

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = (
                            accumulated_loss / self.config.gradient_accumulation_steps
                        )
                        lr = self.scheduler.get_last_lr()[0]
                        
                        # Calculate throughput
                        samples_per_step = self.config.batch_size * self.config.gradient_accumulation_steps
                        throughput = samples_per_step / t_step if t_step > 0 else 0
                        
                        # GPU memory
                        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                        self._log_metrics(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch
                                + batch_idx / len(self.train_dataloader),
                                # Timing metrics
                                "train/t_data": timing_accum["t_data"],
                                "train/t_transfer": timing_accum["t_transfer"],
                                "train/t_forward": timing_accum["t_forward"],
                                "train/t_backward": timing_accum["t_backward"],
                                "train/t_optimizer": t_optimizer,
                                "train/t_step": t_step,
                                "train/throughput": throughput,
                                "train/gpu_mem_gb": gpu_mem_gb,
                            },
                            step=self.global_step,
                        )

                        progress_bar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "lr": f"{lr:.2e}",
                                "t/step": f"{t_step:.1f}s",
                                "samp/s": f"{throughput:.1f}",
                            }
                        )

                    # Reset accumulation and timing
                    accumulated_loss = 0.0
                    step_in_accumulation = 0
                    timing_accum = {"t_data": 0.0, "t_transfer": 0.0, "t_forward": 0.0, "t_backward": 0.0}
                    step_start_time = time.perf_counter()
                    progress_bar.update(1)

                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics, step=self.global_step)

                        # Save best model
                        if eval_metrics["eval/loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["eval/loss"]
                            self._save_checkpoint("best")

                        self.model.train()

                    # Regular checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(f"step_{self.global_step}")

                    # Check max steps
                    if (
                        self.config.max_steps
                        and self.global_step >= self.config.max_steps
                    ):
                        logger.info(f"Reached max_steps ({self.config.max_steps})")
                        break

            # End of epoch
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        progress_bar.close()

        # Final save
        self._save_checkpoint("final")

        logger.info("Training complete!")
        logger.info(f"Best eval loss: {self.best_eval_loss:.4f}")

        if self.wandb_logger:
            self.wandb_logger.finish()

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Execute single training step.

        Args:
            batch: Batch dictionary with model inputs

        Returns:
            Loss tensor
        """
        # Use bfloat16 autocast if available
        dtype = getattr(torch, self.config.model.torch_dtype)

        with torch.amp.autocast(
            device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)
        ):
            outputs = self.model.base_model.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                image_flags=batch["image_flags"],
                labels=batch["labels"],
            )
            loss = outputs.loss

        return loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on validation set.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation...")
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        dtype = getattr(torch, self.config.model.torch_dtype)

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = self._to_device(batch)

            with torch.amp.autocast(
                device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)
            ):
                outputs = self.model.base_model.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    image_flags=batch["image_flags"],
                    labels=batch["labels"],
                )

            batch_size = batch["input_ids"].shape[0]
            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        metrics = {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
            "eval/samples": total_samples,
        }

        logger.info(f"Eval loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

        return metrics

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to console, CSV, and WandB."""
        # Add step to metrics for CSV logger
        metrics_with_step = {"step": step, **metrics}
        
        # Console logging
        metrics_str = ", ".join(
            f"{k}: {v:.2e}" for k, v in metrics.items() if isinstance(v, (int, float))
        )
        logger.info(f"Step {step}: {metrics_str}")

        # CSV logging
        if self.csv_logger:
            # Determine if this is train or eval metrics
            if any(k.startswith("train/") for k in metrics):
                self.csv_logger.log_train(metrics_with_step)
            elif any(k.startswith("eval/") for k in metrics):
                self.csv_logger.log_eval(metrics_with_step)

        # WandB logging
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)

    def _save_checkpoint(self, name: str):
        """
        Save training checkpoint.

        Args:
            name: Checkpoint name (e.g., "best", "step_1000", "final")
        """
        checkpoint_dir = self.exp_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        save_lora_weights(self.model, str(checkpoint_dir))

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(state, checkpoint_dir / "training_state.pt")

        logger.info(f"Saved checkpoint: {checkpoint_dir}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_dir = Path(checkpoint_path)

        # Load LoRA weights first (before optimizer, since optimizer references model params)
        adapter_path = checkpoint_dir / "adapter"
        if adapter_path.exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            
            # Load adapter weights - try safetensors first, then pytorch
            adapter_weights_path = adapter_path / "adapter_model.safetensors"
            if adapter_weights_path.exists():
                adapter_state_dict = load_file(str(adapter_weights_path))
            else:
                adapter_weights_path = adapter_path / "adapter_model.bin"
                if adapter_weights_path.exists():
                    adapter_state_dict = torch.load(adapter_weights_path, map_location=self.device)
                else:
                    logger.warning(f"No adapter weights found in {adapter_path}")
                    adapter_state_dict = None
            
            if adapter_state_dict is not None:
                set_peft_model_state_dict(self.model, adapter_state_dict)
                logger.info(f"Loaded adapter weights from {adapter_path}")

        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)

            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_eval_loss = state["best_eval_loss"]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

            if self.scaler is not None and "scaler_state_dict" in state:
                self.scaler.load_state_dict(state["scaler_state_dict"])

            logger.info(
                f"Resumed from checkpoint: step={self.global_step}, epoch={self.epoch}"
            )

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints_dir = self.exp_dir / "checkpoints"

        # Get step-based checkpoints (not 'best' or 'final')
        step_checkpoints = []
        for d in checkpoints_dir.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                try:
                    step = int(d.name.split("_")[1])
                    step_checkpoints.append((step, d))
                except ValueError:
                    continue

        # Sort by step number
        step_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove old checkpoints
        for _, checkpoint_dir in step_checkpoints[self.config.save_total_limit :]:
            import shutil

            shutil.rmtree(checkpoint_dir)
            logger.info(f"Removed old checkpoint: {checkpoint_dir}")