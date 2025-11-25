"""DataLoader for RoboVQA preprocessed data."""

import json
import logging
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class RoboVQADataset(Dataset):
    """
    Dataset for preprocessed RoboVQA data.

    Loads JSONL files with format:
    {
        "id": "video_id_qa_idx",
        "images": ["images/xxx_00.jpg", ..., "images/xxx_15.jpg"],
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ],
        "metadata": {"task_type": "...", ...}
    }
    """

    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        processor: Optional[Callable] = None,
        max_frames: int = 16,
        subset_ratio: float = 1.0,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize RoboVQA dataset.

        Args:
            jsonl_path: Path to JSONL file (train.jsonl or val.jsonl)
            images_dir: Path to base images directory (parent of images/)
            processor: Optional processor/tokenizer for the model
            max_frames: Maximum frames to use per sample (default 16)
            subset_ratio: Fraction of data to use (0.0-1.0)
            seed: Random seed for subset sampling
            max_samples: Hard cap on number of samples (overrides subset_ratio)
        """
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_frames = max_frames

        # Load samples from JSONL
        logger.info(f"Loading dataset from {jsonl_path}")
        self.samples = self._load_jsonl(jsonl_path)
        total_samples = len(self.samples)

        # Apply subset sampling
        if max_samples is not None:
            n_samples = min(max_samples, total_samples)
        else:
            n_samples = int(total_samples * subset_ratio)

        if n_samples < total_samples:
            random.seed(seed)
            self.samples = random.sample(self.samples, n_samples)
            logger.info(
                f"Subsampled to {n_samples}/{total_samples} samples "
                f"({100*n_samples/total_samples:.1f}%)"
            )

        # Validate sample structure
        self._validate_samples()

        # Compute task distribution
        self.task_distribution = self._compute_task_distribution()
        logger.info(f"Task distribution: {self.task_distribution}")

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load samples from JSONL file."""
        samples = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _validate_samples(self):
        """Validate sample structure and log statistics."""
        required_keys = {"id", "images", "conversations"}

        valid_samples = []
        for sample in self.samples:
            # Check required keys
            if not required_keys.issubset(sample.keys()):
                missing = required_keys - set(sample.keys())
                logger.warning(
                    f"Sample {sample.get('id', 'unknown')} missing keys: {missing}"
                )
                continue

            # Check conversations format
            convs = sample.get("conversations", [])
            if len(convs) < 2:
                logger.warning(f"Sample {sample['id']} has insufficient conversations")
                continue

            # Check first image exists (spot check)
            images = sample.get("images", [])
            if images:
                # Images paths are relative to data_dir (e.g., "images/xxx.jpg")
                first_image = images[0]
                # Remove "images/" prefix if present since images_dir already points to images/
                if first_image.startswith("images/"):
                    first_image = first_image[7:]  # Remove "images/" prefix
                frame_path = self.images_dir / first_image
                if not frame_path.exists():
                    logger.warning(f"Frame not found: {frame_path}")
                    continue

            valid_samples.append(sample)

        if len(valid_samples) < len(self.samples):
            logger.warning(
                f"Filtered {len(self.samples) - len(valid_samples)} invalid samples"
            )

        self.samples = valid_samples
        logger.info(f"Loaded {len(self.samples)} valid samples")

    def _compute_task_distribution(self) -> Dict[str, int]:
        """Compute distribution of task types."""
        distribution = {}
        for sample in self.samples:
            task = sample.get("metadata", {}).get("task_type", "unknown")
            distribution[task] = distribution.get(task, 0) + 1
        return distribution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dict with keys:
                - images: List of PIL Images
                - question: str (human turn)
                - answer: str (gpt turn)
                - metadata: Dict with id, task_type, etc.
        """
        sample = self.samples[idx]

        # Load frames (limit to max_frames)
        image_paths = sample["images"][: self.max_frames]
        images = []
        for img_path in image_paths:
            # Remove "images/" prefix if present
            if img_path.startswith("images/"):
                img_path = img_path[7:]
            frame_path = self.images_dir / img_path
            try:
                img = Image.open(frame_path).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to load {frame_path}: {e}")
                # Create placeholder black image
                images.append(Image.new("RGB", (448, 448), color="black"))

        # Pad if fewer frames than expected
        while len(images) < self.max_frames:
            images.append(images[-1] if images else Image.new("RGB", (448, 448)))

        # Extract question/answer from conversations
        convs = sample["conversations"]
        question = ""
        answer = ""
        for conv in convs:
            if conv["from"] == "human":
                question = conv["value"]
            elif conv["from"] == "gpt":
                answer = conv["value"]

        result = {
            "images": images,
            "question": question,
            "answer": answer,
            "metadata": {
                "id": sample["id"],
                "task_type": sample.get("metadata", {}).get("task_type", "unknown"),
            },
        }

        # Apply processor if provided
        if self.processor is not None:
            result = self.processor(result)

        return result


class InternVLCollator:
    """
    Collator for InternVL3 that handles variable-length sequences.

    Formats data for InternVL3's chat template with image tokens.
    Uses standard torchvision transforms for image preprocessing.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        image_size: int = 448,  # InternVL3 default
        num_image_tokens: int = 256,  # InternVL3 uses 256 tokens per tile
        max_dynamic_patch: int = 1,  # Tiles per image (1 = no dynamic patching)
        img_context_token_id: int = 151667,  # InternVL3 default, should come from model
    ):
        """
        Initialize collator.

        Args:
            tokenizer: InternVL3 tokenizer
            max_length: Maximum sequence length
            image_size: Image size for InternVL3 (default 448)
            num_image_tokens: Number of tokens per tile in InternVL3
            max_dynamic_patch: Maximum tiles per image (1 for fixed resolution)
            img_context_token_id: Token ID for image context (from model config)
        """
        import torchvision.transforms as T

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_image_tokens = num_image_tokens
        self.max_dynamic_patch = max_dynamic_patch
        self.img_context_token_id = img_context_token_id
        # Tokens per image = tiles_per_image * tokens_per_tile
        self.tokens_per_image = max_dynamic_patch * num_image_tokens

        # Standard InternVL3 image preprocessing
        self.image_transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch for InternVL3."""
        batch_images = []
        batch_input_ids = []
        batch_labels = []

        for item in batch:
            images = item["images"]
            question = item["question"]  # Contains <image> markers
            answer = item["answer"]

            num_images = len(images)  # Actual number of images loaded

            # Process images
            processed_images = [self.image_transform(img) for img in images]
            pixel_values = torch.stack(processed_images, dim=0)
            batch_images.append(pixel_values)

            # Build input_ids with proper <IMG_CONTEXT> tokens
            prompt = f"{question}\nAnswer:"
            full_text = f"{prompt} {answer}"

            # Split by <image> and only use num_images worth of markers
            segments = full_text.split("<image>")

            # Reconstruct text with only num_images markers
            # segments[0] + <image> + segments[1] + <image> + ... + segments[num_images] + remaining_segments_joined
            if len(segments) > num_images + 1:
                # More <image> markers than images - truncate
                kept_segments = segments[: num_images + 1]
                # Join remaining segments without <image> between them
                remaining = "".join(segments[num_images + 1 :])
                kept_segments[-1] = kept_segments[-1] + remaining
                segments = kept_segments

            input_ids = []

            for i, segment in enumerate(segments):
                if i > 0 and i <= num_images:
                    # Insert tokens_per_image IMG_CONTEXT tokens for each actual image
                    input_ids.extend(
                        [self.img_context_token_id] * self.tokens_per_image
                    )

                if segment:
                    tokens = self.tokenizer.encode(segment, add_special_tokens=False)
                    input_ids.extend(tokens)

            # Add EOS
            input_ids.append(self.tokenizer.eos_token_id)

            # Calculate where prompt ends for label masking
            prompt_for_mask = f"{question}\nAnswer:"
            prompt_segments = prompt_for_mask.split("<image>")
            # Same truncation logic for prompt
            if len(prompt_segments) > num_images + 1:
                kept_segments = prompt_segments[: num_images + 1]
                remaining = "".join(prompt_segments[num_images + 1 :])
                kept_segments[-1] = kept_segments[-1] + remaining
                prompt_segments = kept_segments

            prompt_len = 0
            for i, segment in enumerate(prompt_segments):
                if i > 0 and i <= num_images:
                    prompt_len += self.tokens_per_image
                if segment:
                    prompt_len += len(
                        self.tokenizer.encode(segment, add_special_tokens=False)
                    )

            # Create labels (mask prompt with -100)
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            batch_input_ids.append(torch.tensor(input_ids))
            batch_labels.append(torch.tensor(labels))

        # Stack images: (B, N, C, H, W) -> (B*N, C, H, W)
        pixel_values = torch.stack(batch_images, dim=0)
        batch_size, num_frames = pixel_values.shape[:2]
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        # image_flags: one entry per tile in pixel_values (derived from actual shape)
        image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long)

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Truncate if needed
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, : self.max_length]
            labels = labels[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
            "labels": labels,
        }


def create_dataloader(
    jsonl_path: str,
    images_dir: str,
    tokenizer=None,
    model=None,  # Model for getting image size config
    batch_size: int = 4,
    subset_ratio: float = 1.0,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    max_frames: int = 16,
    max_length: int = 2048,
    max_dynamic_patch: int = 1,
    seed: int = 42,
    image_size: int = 448,  # Default InternVL3 image size
) -> DataLoader:
    """
    Create DataLoader for RoboVQA training.

    Args:
        jsonl_path: Path to JSONL file
        images_dir: Path to images directory
        tokenizer: Model tokenizer (optional, for raw data)
        model: Model instance (used for config, optional)
        batch_size: Batch size
        subset_ratio: Fraction of data to use
        max_samples: Maximum samples (overrides subset_ratio)
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        max_frames: Maximum frames per sample
        max_length: Maximum sequence length
        seed: Random seed
        image_size: Image size for preprocessing (default 448)

    Returns:
        DataLoader instance
    """
    dataset = RoboVQADataset(
        jsonl_path=jsonl_path,
        images_dir=images_dir,
        max_frames=max_frames,
        subset_ratio=subset_ratio,
        max_samples=max_samples,
        seed=seed,
    )

    # Create collator if tokenizer provided
    collate_fn = None
    if tokenizer is not None:
        # Try to get config values from model
        img_context_token_id = 151667  # Default for InternVL3
        if model is not None:
            # Try to get image size from model config
            if hasattr(model, "config"):
                try:
                    image_size = getattr(model.config, "force_image_size", image_size)
                except:
                    pass
            # Get img_context_token_id from model (set in internvl.py)
            # Check both wrapped and base model for PEFT compatibility
            if hasattr(model, "img_context_token_id"):
                img_context_token_id = model.img_context_token_id
            elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                base = model.base_model.model
                if hasattr(base, "img_context_token_id"):
                    img_context_token_id = base.img_context_token_id

        collate_fn = InternVLCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            image_size=image_size,
            max_dynamic_patch=max_dynamic_patch,
            img_context_token_id=img_context_token_id,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Avoid batch size issues with DDP
    )

    return dataloader


# Convenience function for stage-based loading
def load_stage_data(
    data_dir: str,
    stage: int,
    split: str = "train",
    max_dynamic_patch: int = 1,
    **kwargs,
) -> DataLoader:
    """
    Load data for a specific training stage.

    Args:
        data_dir: Base data directory (e.g., data/robovqa/processed)
        stage: Training stage (1 or 2)
        split: Data split ("train" or "val")
        **kwargs: Additional arguments for create_dataloader

    Returns:
        DataLoader for the specified stage/split
    """
    data_dir = Path(data_dir)
    jsonl_path = data_dir / f"stage{stage}" / f"{split}.jsonl"
    images_dir = data_dir / "images"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    logger.info(f"Loading Stage {stage} {split} data from {jsonl_path}")

    return create_dataloader(
        jsonl_path=str(jsonl_path),
        images_dir=str(images_dir),
        max_dynamic_patch=max_dynamic_patch,
        **kwargs,
    )
