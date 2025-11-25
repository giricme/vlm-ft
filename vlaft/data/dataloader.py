"""DataLoader for RoboVQA preprocessed data."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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
            logger.info(f"Subsampled to {n_samples}/{total_samples} samples "
                       f"({100*n_samples/total_samples:.1f}%)")
        
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
                logger.warning(f"Sample {sample.get('id', 'unknown')} missing keys: {missing}")
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
            logger.warning(f"Filtered {len(self.samples) - len(valid_samples)} invalid samples")
        
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
        image_paths = sample["images"][:self.max_frames]
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
            }
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
        num_image_tokens: int = 256,  # InternVL3 uses 256 tokens per image
    ):
        """
        Initialize collator.
        
        Args:
            tokenizer: InternVL3 tokenizer
            max_length: Maximum sequence length
            image_size: Image size for InternVL3 (default 448)
            num_image_tokens: Number of tokens per image in InternVL3
        """
        import torchvision.transforms as T
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_image_tokens = num_image_tokens
        
        # Standard InternVL3 image preprocessing
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch for InternVL3.
        
        InternVL3 expects:
        - pixel_values: (B, N, C, H, W) where N is number of images
        - input_ids: (B, seq_len) with <IMG_CONTEXT> tokens
        - attention_mask: (B, seq_len)
        - labels: (B, seq_len) with -100 for non-answer tokens
        """
        batch_images = []
        batch_texts = []
        batch_answers = []
        
        for item in batch:
            images = item["images"]  # List of PIL Images
            question = item["question"]  # Already contains <image> tokens
            answer = item["answer"]
            
            # Process images with torchvision transforms
            processed_images = [self.image_transform(img) for img in images]
            pixel_values = torch.stack(processed_images, dim=0)  # (N, C, H, W)
            batch_images.append(pixel_values)
            
            # Question already has <image> tokens from preprocessing
            # Format: "<image>\n<image>\n...<image>\n[question text]"
            prompt = f"{question}\nAnswer:"
            
            batch_texts.append(prompt)
            batch_answers.append(answer)
        
        # Stack images: (B, N, C, H, W)
        pixel_values = torch.stack(batch_images, dim=0)
        
        # Tokenize prompts + answers for training
        full_texts = [f"{p} {a}" for p, a in zip(batch_texts, batch_answers)]
        
        # Tokenize
        encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        
        # Create labels (mask prompt tokens with -100)
        labels = input_ids.clone()
        
        # Find where answer starts for each sample
        for i, (prompt, answer) in enumerate(zip(batch_texts, batch_answers)):
            prompt_tokens = self.tokenizer(
                prompt, 
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids
            prompt_len = prompt_tokens.shape[1]
            
            # Mask prompt tokens
            labels[i, :prompt_len] = -100
        
        # Also mask padding
        labels[attention_mask == 0] = -100

        # Flatten for InternVL3: (B, N, C, H, W) -> (B*N, C, H, W)
        batch_size, num_frames = pixel_values.shape[:2]
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        image_flags = torch.ones(batch_size * num_frames, dtype=torch.long)
        
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
        # Try to get image size from model config
        if model is not None and hasattr(model, 'config'):
            try:
                # InternVL3 stores image size in vision config
                image_size = getattr(model.config, 'force_image_size', image_size)
            except:
                pass
        
        collate_fn = InternVLCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            image_size=image_size,
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
        **kwargs,
    )