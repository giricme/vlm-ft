"""Logging utilities for experiment tracking and monitoring."""

from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import yaml


def setup_logging(
    experiment_dir: Path,
    experiment_name: Optional[str] = None,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up logging with file and console handlers.

    Args:
        experiment_dir: Directory to save log files
        experiment_name: Experiment name to prefix log files (optional)
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_file:
        # Save to logs/ folder with experiment name prefix
        log_filename = (
            f"{experiment_name}_training.log" if experiment_name else "training.log"
        )
        log_file = experiment_dir / "logs" / log_filename
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def create_experiment_dir(
    base_dir: str = "experiments",
    env_name: str = "cartpole",
    agent_type: str = "dqn",
    seed: Optional[int] = None,
    suffix: Optional[str] = None,
) -> Path:
    """
    Create experiment directory with timestamp and metadata.

    Args:
        base_dir: Base directory for all experiments
        env_name: Environment name (e.g., 'cartpole', 'pong')
        agent_type: Agent type (e.g., 'dqn', 'lora_option1')
        seed: Random seed for the run
        suffix: Optional suffix for directory name

    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name_parts = [env_name, agent_type, timestamp]

    if seed is not None:
        dir_name_parts.append(f"seed{seed}")

    if suffix:
        dir_name_parts.append(suffix)

    dir_name = "_".join(dir_name_parts)
    exp_dir = Path(base_dir) / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "videos").mkdir(exist_ok=True)

    return exp_dir


def save_config(
    config: Dict[str, Any], experiment_dir: Path, experiment_name: Optional[str] = None
) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        experiment_dir: Directory to save config
        experiment_name: Experiment name to prefix config file (optional)
    """
    config_filename = (
        f"{experiment_name}_config.yaml" if experiment_name else "config.yaml"
    )
    config_file = experiment_dir / config_filename
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class WandBLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        mode: str = "offline",  # Changed default to offline
        enabled: bool = True,  # New parameter to completely disable wandb
    ):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            entity: WandB entity (username or team)
            config: Configuration dictionary to log
            name: Run name
            tags: List of tags
            notes: Notes about the run
            group: Group name for organizing runs
            job_type: Job type (e.g., 'train', 'eval')
            mode: 'online', 'offline', or 'disabled'
            enabled: Whether to enable wandb at all
        """
        self.enabled = enabled and mode != "disabled"
        self.wandb = None
        self.run = None

        if not self.enabled:
            logging.getLogger("loradqn").info("WandB logging disabled")
            return

        try:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                entity=entity,
                config=config,
                name=name,
                tags=tags,
                notes=notes,
                group=group,
                job_type=job_type,
                mode=mode,
            )
            logging.getLogger("loradqn").info(f"WandB initialized in {mode} mode")

        except ImportError:
            self.enabled = False
            logging.getLogger("loradqn").warning(
                "wandb not installed. Run: pip install wandb. Continuing without wandb."
            )
        except Exception as e:
            self.enabled = False
            logging.getLogger("loradqn").warning(
                f"Failed to initialize wandb: {e}. Continuing without wandb."
            )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.enabled and self.run is not None:
            self.wandb.log(metrics, step=step)

    def log_video(self, video_path: str, step: Optional[int] = None) -> None:
        """
        Log video to WandB.

        Args:
            video_path: Path to video file
            step: Optional step number
        """
        if self.enabled and self.run is not None:
            self.wandb.log({"video": self.wandb.Video(video_path)}, step=step)

    def finish(self) -> None:
        """Finish the WandB run."""
        if self.enabled and self.run is not None:
            self.run.finish()

    def save_model(self, model_path: str, aliases: Optional[list] = None) -> None:
        """
        Save model checkpoint to WandB.

        Args:
            model_path: Path to model file
            aliases: Optional list of aliases (e.g., ['latest', 'best'])
        """
        if self.enabled and self.run is not None:
            artifact = self.wandb.Artifact(
                name=f"model-{self.run.id}",
                type="model",
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact, aliases=aliases)


class CSVLogger:
    """Log metrics to CSV files for easy analysis."""

    def __init__(
        self, log_dir: Path, experiment_name: Optional[str] = None, enabled: bool = True
    ):
        """
        Initialize CSV logger.

        Args:
            log_dir: Directory to save CSV files
            experiment_name: Experiment name to prefix CSV files (optional)
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        if not enabled:
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Prefix CSV files with experiment name
        train_filename = (
            f"{experiment_name}_train_metrics.csv"
            if experiment_name
            else "train_metrics.csv"
        )
        eval_filename = (
            f"{experiment_name}_eval_metrics.csv"
            if experiment_name
            else "eval_metrics.csv"
        )

        self.train_file = self.log_dir / train_filename
        self.eval_file = self.log_dir / eval_filename

        # Track headers dynamically for eval (to support action distribution)
        self.eval_headers = self._get_eval_headers()
        self.eval_headers_written = False

        # Initialize files with headers
        self._init_csv(self.train_file, self._get_train_headers())
        # Don't init eval file yet - will do on first log with dynamic headers

        logging.getLogger("loradqn").info(f"CSV logging enabled: {self.log_dir}")

    def _get_train_headers(self) -> List[str]:
        """Get training CSV headers."""
        return [
            # Basic metrics
            "step",
            "timestamp",
        ]

    def _get_eval_headers(self) -> List[str]:
        """Get evaluation CSV headers."""
        return [
            "step",
            "timestamp",
        ]

    def _init_csv(self, filepath: Path, headers: List[str]):
        """Initialize CSV file with headers."""
        if not filepath.exists():
            with open(filepath, "w") as f:
                f.write(",".join(headers) + "\n")

    def log_train(self, metrics: Dict[str, Any]):
        """
        Log training metrics.

        Args:
            metrics: Dictionary with training metrics
        """
        if not self.enabled:
            return

        from datetime import datetime

        # Helper to get metric with multiple possible keys
        def get_metric(*keys):
            for key in keys:
                if key in metrics and metrics[key] != "":
                    return metrics[key]
            return ""

        # Extract and format metrics
        row = {
            "step": metrics.get("step", ""),
            "loss": metrics.get("train/loss", ""),
            "timestamp": datetime.now().isoformat(),
        }

        self._append_row(self.train_file, row, self._get_train_headers())

    def log_eval(self, metrics: Dict[str, Any]):
        """
        Log evaluation metrics with dynamic column support for action distributions.

        Args:
            metrics: Dictionary with evaluation metrics
        """
        if not self.enabled:
            return

        from datetime import datetime

        # Extract and format metrics
        row = {
            "step": metrics.get("step", ""),
            "timestamp": datetime.now().isoformat(),
        }

        # Add any action distribution columns dynamically
        for key, value in metrics.items():
            if key.startswith("action_") and key not in row:
                row[key] = value

        # On first write, determine final headers from actual data
        if not self.eval_headers_written:
            # Add any action_ columns to headers
            action_cols = sorted([k for k in row.keys() if k.startswith("action_")])
            self.eval_headers = self._get_eval_headers() + action_cols
            self._init_csv(self.eval_file, self.eval_headers)
            self.eval_headers_written = True

        self._append_row(self.eval_file, row, self.eval_headers)

    def _append_row(self, filepath: Path, row: Dict[str, Any], headers: List[str]):
        """Append a row to CSV file."""
        with open(filepath, "a") as f:
            values = [str(row.get(h, "")) for h in headers]
            f.write(",".join(values) + "\n")


class MetricsTracker:
    """Track and compute running statistics for metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.counts = {}

    def update(self, metric_name: str, value: float) -> None:
        """
        Update running statistics for a metric.

        Args:
            metric_name: Name of the metric
            value: New value to add
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0.0, "count": 0, "values": []}

        self.metrics[metric_name]["sum"] += value
        self.metrics[metric_name]["count"] += 1
        self.metrics[metric_name]["values"].append(value)

    def get_mean(self, metric_name: str) -> Optional[float]:
        """
        Get mean value of a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Mean value or None if metric doesn't exist
        """
        if metric_name not in self.metrics or self.metrics[metric_name]["count"] == 0:
            return None
        return self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]

    def get_last(self, metric_name: str) -> Optional[float]:
        """
        Get last value of a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Last value or None if metric doesn't exist
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]["values"]:
            return None
        return self.metrics[metric_name]["values"][-1]

    def reset(self, metric_name: Optional[str] = None) -> None:
        """
        Reset metrics.

        Args:
            metric_name: Name of specific metric to reset, or None to reset all
        """
        if metric_name is None:
            self.metrics.clear()
        elif metric_name in self.metrics:
            self.metrics[metric_name] = {"sum": 0.0, "count": 0, "values": []}

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary of metric names to mean values
        """
        return {name: self.get_mean(name) for name in self.metrics.keys()}
