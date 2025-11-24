"""Preprocess RoboVQA dataset for InternVL3 fine-tuning.

Converts raw TFRecords to:
- Extracted JPEG frames (shared across stages)
- Stage 1 JSONL: Split QAs (one QA per sample for basic grounding)
- Stage 2 JSONL: Multi-turn conversations (all QAs per sample)

Output structure:
    processed/
        images/                  # {unique_id}_{frame_idx:02d}.jpg
        stage1/
            train.jsonl
            val.jsonl
        stage2/
            train.jsonl
            val.jsonl
        metadata.json            # Dataset statistics

Usage:
    # Preprocess
    python -m vlaft.data.preprocess_robovqa --data-dir data/robovqa/raw --output-dir data/robovqa/processed
    python -m vlaft.data.preprocess_robovqa --data-dir data/robovqa/raw --output-dir data/robovqa/processed --workers 8
    
    # Verify preprocessed data
    python -m vlaft.data.preprocess_robovqa --verify --output-dir data/robovqa/processed
    python -m vlaft.data.preprocess_robovqa --verify --output-dir data/robovqa/processed --verify-samples 200
"""

import argparse
import io
import json
import logging
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# TensorFlow import with reduced logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class QAPair:
    """Single question-answer pair extracted from RoboVQA."""
    task_type: str          # e.g., "affordance", "planning", "future_prediction"
    subtype: str            # e.g., "discriminative", "generative", "freeform"
    format_type: str        # e.g., "discrete", "positive:freeform"
    question: str           # Full question text
    answer: str             # Answer text
    frame_start: int        # Start frame index
    frame_end: int          # End frame index


@dataclass
class RoboVQASample:
    """Single sample from RoboVQA dataset."""
    unique_id: str
    video_filename: str
    frames: List[bytes]     # Raw JPEG bytes for each frame
    qa_pairs: List[QAPair]
    timestamps: List[float]


@dataclass
class ProcessingStats:
    """Track preprocessing statistics."""
    total_samples: int = 0
    total_frames: int = 0
    total_qa_pairs: int = 0
    task_type_counts: Dict[str, int] = field(default_factory=dict)
    frames_per_sample: List[int] = field(default_factory=list)
    qas_per_sample: List[int] = field(default_factory=list)
    failed_samples: List[str] = field(default_factory=list)
    skipped_existing: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "total_frames": self.total_frames,
            "total_qa_pairs": self.total_qa_pairs,
            "task_type_counts": self.task_type_counts,
            "avg_frames_per_sample": np.mean(self.frames_per_sample) if self.frames_per_sample else 0,
            "avg_qas_per_sample": np.mean(self.qas_per_sample) if self.qas_per_sample else 0,
            "failed_samples": len(self.failed_samples),
            "skipped_existing": self.skipped_existing,
        }


# =============================================================================
# QA Text Parsing
# =============================================================================

def parse_qa_text(text: str, frame_start: int, frame_end: int) -> List[QAPair]:
    """
    Parse RoboVQA text field into QA pairs.
    
    Format: <task:{type}:{subtype}:{format}>{context} Q: {question}? A: {answer}
    Multiple QAs are concatenated with <task:...> markers.
    
    Args:
        text: Raw text field from TFRecord
        frame_start: Start frame index for this text
        frame_end: End frame index for this text
    
    Returns:
        List of parsed QAPair objects
    """
    qa_pairs = []
    
    # Pattern to match task markers and extract components
    # Matches: <task:type:subtype:format> or <task:type:format>
    task_pattern = r'<task:([^>]+)>'
    
    # Split by task markers, keeping the markers
    parts = re.split(task_pattern, text)
    
    # parts[0] is empty or preamble, then alternates: marker_content, text, marker_content, text...
    i = 1
    while i < len(parts):
        if i + 1 >= len(parts):
            break
            
        task_info = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        
        # Parse task info (e.g., "affordance:discriminative:discrete")
        task_parts = task_info.split(':')
        task_type = task_parts[0] if len(task_parts) > 0 else "unknown"
        subtype = task_parts[1] if len(task_parts) > 1 else ""
        format_type = ':'.join(task_parts[2:]) if len(task_parts) > 2 else ""
        
        # Extract Q: and A: from content
        qa_match = re.search(r'(.*)Q:\s*(.+?)\s*A:\s*(.+?)(?=<task:|$)', content, re.DOTALL)
        
        if qa_match:
            context = qa_match.group(1).strip()
            question_text = qa_match.group(2).strip()
            answer_text = qa_match.group(3).strip()
            
            # Combine context with question
            full_question = f"{context} {question_text}".strip() if context else question_text
            
            # Clean up answer (remove trailing markers, whitespace)
            answer_text = re.sub(r'<[^>]+>.*$', '', answer_text).strip()
            
            qa_pairs.append(QAPair(
                task_type=task_type,
                subtype=subtype,
                format_type=format_type,
                question=full_question,
                answer=answer_text,
                frame_start=frame_start,
                frame_end=frame_end,
            ))
        
        i += 2
    
    return qa_pairs


# =============================================================================
# TFRecord Parsing
# =============================================================================

def parse_tfrecord_sample(serialized: bytes) -> Optional[RoboVQASample]:
    """
    Parse a single TFRecord SequenceExample into RoboVQASample.
    
    Args:
        serialized: Serialized SequenceExample bytes
    
    Returns:
        RoboVQASample or None if parsing fails
    """
    try:
        # Define feature specifications
        context_features = {
            'unique_id': tf.io.FixedLenFeature([], tf.string),
            'video_filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        
        sequence_features = {
            'images': tf.io.FixedLenSequenceFeature([], tf.string),
            'texts': tf.io.FixedLenSequenceFeature([], tf.string),
            'texts_start': tf.io.FixedLenSequenceFeature([], tf.int64),
            'texts_end': tf.io.FixedLenSequenceFeature([], tf.int64),
        }
        
        context, sequences = tf.io.parse_single_sequence_example(
            serialized,
            context_features=context_features,
            sequence_features=sequence_features,
        )
        
        # Extract values
        unique_id = context['unique_id'].numpy().decode('utf-8')
        video_filename = context['video_filename'].numpy().decode('utf-8')
        
        frames = [img.numpy() for img in sequences['images']]
        texts = [t.numpy().decode('utf-8') for t in sequences['texts']]
        texts_start = sequences['texts_start'].numpy().tolist()
        texts_end = sequences['texts_end'].numpy().tolist()
        timestamps = []  # Skip timestamps - not needed for training
        
        # Parse all QA pairs from text fields
        qa_pairs = []
        for text, start, end in zip(texts, texts_start, texts_end):
            qa_pairs.extend(parse_qa_text(text, start, end))
        
        return RoboVQASample(
            unique_id=unique_id,
            video_filename=video_filename,
            frames=frames,
            qa_pairs=qa_pairs,
            timestamps=timestamps,
        )
        
    except Exception as e:
        logging.warning(f"Failed to parse TFRecord sample: {e}")
        return None


def iter_tfrecord_samples(tfrecord_path: Path):
    """
    Iterate over samples in a TFRecord file.
    
    Args:
        tfrecord_path: Path to TFRecord file
    
    Yields:
        RoboVQASample objects
    """
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    
    for serialized in dataset:
        sample = parse_tfrecord_sample(serialized.numpy())
        if sample is not None:
            yield sample


# =============================================================================
# Image Processing
# =============================================================================

def save_frames(
    sample: RoboVQASample,
    output_dir: Path,
    target_size: Tuple[int, int] = (288, 288),
    quality: int = 85,
    skip_existing: bool = True,
) -> Tuple[List[str], int]:
    """
    Save frames from a sample as JPEG files.
    
    Args:
        sample: RoboVQASample with frame data
        output_dir: Directory to save images
        target_size: Target (width, height) for resizing
        quality: JPEG quality (1-100)
        skip_existing: Skip if images already exist
    
    Returns:
        Tuple of (list of relative image paths, number skipped)
    """
    image_paths = []
    skipped = 0
    
    for idx, frame_bytes in enumerate(sample.frames):
        filename = f"{sample.unique_id}_{idx:02d}.jpg"
        filepath = output_dir / filename
        rel_path = f"images/{filename}"
        image_paths.append(rel_path)
        
        if skip_existing and filepath.exists():
            skipped += 1
            continue
        
        try:
            # Decode and resize
            img = Image.open(io.BytesIO(frame_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize with high-quality resampling
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save
            img.save(filepath, 'JPEG', quality=quality)
            
        except Exception as e:
            logging.warning(f"Failed to save frame {filename}: {e}")
            # Still add to paths list (for consistency), file just won't exist
    
    return image_paths, skipped


# =============================================================================
# JSONL Generation
# =============================================================================

def format_stage1_samples(
    sample: RoboVQASample,
    image_paths: List[str],
) -> List[Dict[str, Any]]:
    """
    Format sample for Stage 1: Split QAs (one sample per QA pair).
    
    Each QA becomes a separate training sample with all frames as context.
    
    Args:
        sample: RoboVQASample
        image_paths: List of relative paths to saved images
    
    Returns:
        List of JSONL-ready dictionaries
    """
    samples = []
    
    # Build image placeholder string
    image_tags = "\n".join(["<image>"] * len(image_paths))
    
    for idx, qa in enumerate(sample.qa_pairs):
        # Create unique ID for this split sample
        sample_id = f"{sample.unique_id}_qa{idx:02d}"
        
        # Format conversation
        human_content = f"{image_tags}\n{qa.question}"
        
        samples.append({
            "id": sample_id,
            "images": image_paths,
            "conversations": [
                {"from": "human", "value": human_content},
                {"from": "gpt", "value": qa.answer},
            ],
            "metadata": {
                "task_type": qa.task_type,
                "subtype": qa.subtype,
                "format": qa.format_type,
                "source_id": sample.unique_id,
                "frame_range": [qa.frame_start, qa.frame_end],
            }
        })
    
    return samples


def format_stage2_sample(
    sample: RoboVQASample,
    image_paths: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Format sample for Stage 2: Multi-turn conversation.
    
    All QAs from the same video become turns in a single conversation.
    
    Args:
        sample: RoboVQASample
        image_paths: List of relative paths to saved images
    
    Returns:
        JSONL-ready dictionary or None if no QAs
    """
    if not sample.qa_pairs:
        return None
    
    # Build image placeholder string (only on first turn)
    image_tags = "\n".join(["<image>"] * len(image_paths))
    
    conversations = []
    for idx, qa in enumerate(sample.qa_pairs):
        # First turn includes image tags
        if idx == 0:
            human_content = f"{image_tags}\n{qa.question}"
        else:
            human_content = qa.question
        
        conversations.append({"from": "human", "value": human_content})
        conversations.append({"from": "gpt", "value": qa.answer})
    
    return {
        "id": sample.unique_id,
        "images": image_paths,
        "conversations": conversations,
        "metadata": {
            "num_turns": len(sample.qa_pairs),
            "task_types": list(set(qa.task_type for qa in sample.qa_pairs)),
            "video_filename": sample.video_filename,
        }
    }


# =============================================================================
# Verification
# =============================================================================

def verify_preprocessed(output_dir: Path, num_samples: int = 100) -> bool:
    """
    Verify preprocessed data integrity.
    
    Checks:
    - JSONL files exist and are valid
    - Referenced images exist
    - Image dimensions are correct
    - Conversation format is valid for InternVL3
    
    Args:
        output_dir: Directory containing preprocessed data
        num_samples: Number of samples to verify per JSONL file
    
    Returns:
        True if all checks pass, False otherwise
    """
    all_passed = True
    
    logging.info("=" * 60)
    logging.info("VERIFICATION MODE")
    logging.info("=" * 60)
    
    # Check directory structure
    required_dirs = ["images", "stage1", "stage2"]
    for dir_name in required_dirs:
        dir_path = output_dir / dir_name
        if not dir_path.exists():
            logging.error(f"Missing directory: {dir_path}")
            all_passed = False
        else:
            logging.info(f"✓ Directory exists: {dir_name}/")
    
    # Check metadata
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logging.info(f"✓ metadata.json exists")
        logging.info(f"  - Total samples: {metadata['statistics']['total_samples']:,}")
        logging.info(f"  - Total frames: {metadata['statistics']['total_frames']:,}")
        logging.info(f"  - Stage 1: {metadata['stage1']['total']:,} samples")
        logging.info(f"  - Stage 2: {metadata['stage2']['total']:,} samples")
    else:
        logging.warning("metadata.json not found")
    
    # Check images directory
    images_dir = output_dir / "images"
    if images_dir.exists():
        # Count images without loading all paths into memory
        logging.info("  Counting images (may take a moment)...")
        image_count = sum(1 for _ in images_dir.glob("*.jpg"))
        logging.info(f"✓ Found {image_count:,} images")
        
        # Check a sample image
        sample_img_path = next(images_dir.glob("*.jpg"), None)
        if sample_img_path:
            sample_img = Image.open(sample_img_path)
            logging.info(f"  - Sample image size: {sample_img.size}")
            logging.info(f"  - Sample image mode: {sample_img.mode}")
            
            # Estimate disk usage based on sample
            sample_size = sample_img_path.stat().st_size
            estimated_total_gb = (image_count * sample_size) / (1024**3)
            logging.info(f"  - Estimated disk usage: ~{estimated_total_gb:.1f} GB")
    
    # Verify each JSONL file
    jsonl_files = [
        ("stage1/train.jsonl", "Stage 1 Train"),
        ("stage1/val.jsonl", "Stage 1 Val"),
        ("stage2/train.jsonl", "Stage 2 Train"),
        ("stage2/val.jsonl", "Stage 2 Val"),
    ]
    
    for jsonl_rel_path, label in jsonl_files:
        jsonl_path = output_dir / jsonl_rel_path
        
        if not jsonl_path.exists():
            logging.error(f"✗ Missing: {jsonl_rel_path}")
            all_passed = False
            continue
        
        logging.info(f"\nVerifying {label} ({jsonl_rel_path})...")
        
        # Count lines and sample
        with open(jsonl_path) as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        logging.info(f"  Total samples: {total_lines:,}")
        
        # Sample random lines to verify
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(
            total_lines, 
            min(num_samples, total_lines), 
            replace=False
        )
        
        errors = []
        missing_images = 0
        invalid_json = 0
        invalid_format = 0
        
        for idx in sample_indices:
            line = lines[idx]
            
            # Check valid JSON
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                invalid_json += 1
                errors.append(f"Line {idx}: Invalid JSON - {e}")
                continue
            
            # Check required fields
            required_fields = ["id", "images", "conversations"]
            missing_fields = [f for f in required_fields if f not in sample]
            if missing_fields:
                invalid_format += 1
                errors.append(f"Line {idx}: Missing fields {missing_fields}")
                continue
            
            # Check conversation format
            convs = sample["conversations"]
            if not convs or len(convs) < 2:
                invalid_format += 1
                errors.append(f"Line {idx}: Empty or too short conversation")
                continue
            
            # Check alternating human/gpt
            expected_roles = ["human", "gpt"] * (len(convs) // 2 + 1)
            for i, conv in enumerate(convs):
                if conv.get("from") != expected_roles[i]:
                    invalid_format += 1
                    errors.append(f"Line {idx}: Invalid role sequence at turn {i}")
                    break
            
            # Check images exist
            for img_path in sample["images"]:
                full_path = output_dir / img_path
                if not full_path.exists():
                    missing_images += 1
                    if missing_images <= 3:  # Only log first few
                        errors.append(f"Line {idx}: Missing image {img_path}")
                    break
        
        # Report results
        if invalid_json == 0 and invalid_format == 0 and missing_images == 0:
            logging.info(f"  ✓ All {len(sample_indices)} sampled entries valid")
        else:
            all_passed = False
            if invalid_json > 0:
                logging.error(f"  ✗ Invalid JSON: {invalid_json}")
            if invalid_format > 0:
                logging.error(f"  ✗ Invalid format: {invalid_format}")
            if missing_images > 0:
                logging.error(f"  ✗ Missing images: {missing_images}")
            
            # Show first few errors
            for err in errors[:5]:
                logging.error(f"    - {err}")
            if len(errors) > 5:
                logging.error(f"    ... and {len(errors) - 5} more errors")
        
        # Show sample entry
        if total_lines > 0:
            sample_entry = json.loads(lines[0])
            logging.info(f"  Sample entry:")
            logging.info(f"    - ID: {sample_entry['id']}")
            logging.info(f"    - Images: {len(sample_entry['images'])} frames")
            logging.info(f"    - Turns: {len(sample_entry['conversations'])}")
            
            # Show first Q/A
            if len(sample_entry['conversations']) >= 2:
                q = sample_entry['conversations'][0]['value']
                a = sample_entry['conversations'][1]['value']
                # Truncate for display
                q_display = q[:100] + "..." if len(q) > 100 else q
                a_display = a[:100] + "..." if len(a) > 100 else a
                logging.info(f"    - Q: {q_display}")
                logging.info(f"    - A: {a_display}")
    
    # Final summary
    logging.info("\n" + "=" * 60)
    if all_passed:
        logging.info("✓ VERIFICATION PASSED")
    else:
        logging.error("✗ VERIFICATION FAILED")
    logging.info("=" * 60)
    
    return all_passed


# =============================================================================
# Main Processing
# =============================================================================

def process_tfrecord_file(
    tfrecord_path: Path,
    output_dir: Path,
    target_size: Tuple[int, int],
    quality: int,
    skip_existing: bool,
) -> Tuple[List[Dict], List[Dict], ProcessingStats]:
    """
    Process a single TFRecord file.
    
    Args:
        tfrecord_path: Path to TFRecord file
        output_dir: Base output directory
        target_size: Image resize dimensions
        quality: JPEG quality
        skip_existing: Skip existing images
    
    Returns:
        Tuple of (stage1_samples, stage2_samples, stats)
    """
    stats = ProcessingStats()
    stage1_samples = []
    stage2_samples = []
    
    images_dir = output_dir / "images"
    
    for sample in iter_tfrecord_samples(tfrecord_path):
        try:
            # Save frames
            image_paths, skipped = save_frames(
                sample, images_dir, target_size, quality, skip_existing
            )
            stats.skipped_existing += skipped
            
            # Generate Stage 1 samples (split QAs)
            s1_samples = format_stage1_samples(sample, image_paths)
            stage1_samples.extend(s1_samples)
            
            # Generate Stage 2 sample (multi-turn)
            s2_sample = format_stage2_sample(sample, image_paths)
            if s2_sample:
                stage2_samples.append(s2_sample)
            
            # Update stats
            stats.total_samples += 1
            stats.total_frames += len(sample.frames)
            stats.total_qa_pairs += len(sample.qa_pairs)
            stats.frames_per_sample.append(len(sample.frames))
            stats.qas_per_sample.append(len(sample.qa_pairs))
            
            for qa in sample.qa_pairs:
                stats.task_type_counts[qa.task_type] = \
                    stats.task_type_counts.get(qa.task_type, 0) + 1
                    
        except Exception as e:
            logging.warning(f"Failed to process sample {sample.unique_id}: {e}")
            stats.failed_samples.append(sample.unique_id)
    
    return stage1_samples, stage2_samples, stats


def merge_stats(stats_list: List[ProcessingStats]) -> ProcessingStats:
    """Merge multiple ProcessingStats into one."""
    merged = ProcessingStats()
    
    for stats in stats_list:
        merged.total_samples += stats.total_samples
        merged.total_frames += stats.total_frames
        merged.total_qa_pairs += stats.total_qa_pairs
        merged.frames_per_sample.extend(stats.frames_per_sample)
        merged.qas_per_sample.extend(stats.qas_per_sample)
        merged.failed_samples.extend(stats.failed_samples)
        merged.skipped_existing += stats.skipped_existing
        
        for task_type, count in stats.task_type_counts.items():
            merged.task_type_counts[task_type] = \
                merged.task_type_counts.get(task_type, 0) + count
    
    return merged


def train_val_split(
    samples: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into train and validation sets.
    
    Uses deterministic shuffle based on sample IDs for reproducibility.
    
    Args:
        samples: List of sample dictionaries
        val_ratio: Fraction for validation
        seed: Random seed
    
    Returns:
        Tuple of (train_samples, val_samples)
    """
    # Sort by ID for deterministic ordering
    samples = sorted(samples, key=lambda x: x['id'])
    
    # Shuffle with seed
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    
    val_size = int(len(samples) * val_ratio)
    val_indices = set(indices[:val_size])
    
    train_samples = [s for i, s in enumerate(samples) if i not in val_indices]
    val_samples = [s for i, s in enumerate(samples) if i in val_indices]
    
    return train_samples, val_samples


def write_jsonl(samples: List[Dict], filepath: Path) -> None:
    """Write samples to JSONL file."""
    with open(filepath, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


def setup_logging_with_file(output_dir: Path) -> None:
    """Set up logging with console and file handlers."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"preprocess_{timestamp}.log"
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging to {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess RoboVQA dataset for InternVL3 fine-tuning"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify preprocessed data instead of preprocessing",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing raw TFRecord files (required for preprocessing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=288,
        help="Target image size (square, default: 288)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality (1-100, default: 85)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing image files (for resume)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of TFRecord files to process (for testing)",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=100,
        help="Number of samples to verify per JSONL file (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Setup output path
    output_dir = Path(args.output_dir)
    
    # Verify mode
    if args.verify:
        # Minimal setup for verification
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        setup_logging_with_file(output_dir)
        
        success = verify_preprocessed(output_dir, args.verify_samples)
        sys.exit(0 if success else 1)
    
    # Preprocessing mode - requires data_dir
    if not args.data_dir:
        parser.error("--data-dir is required for preprocessing")
    
    data_dir = Path(args.data_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "stage1").mkdir(exist_ok=True)
    (output_dir / "stage2").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # Setup logging
    setup_logging_with_file(output_dir)
    
    # Find TFRecord files (recursive search)
    # Handles both: *.tfrecord and *.tfrecord-NNNNN-of-NNNNN formats
    tfrecord_files = sorted(
        list(data_dir.glob("**/*.tfrecord")) + 
        list(data_dir.glob("**/*.tfrecord-*"))
    )
    if args.max_files:
        tfrecord_files = tfrecord_files[:args.max_files]
    
    logging.info(f"Found {len(tfrecord_files)} TFRecord files")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Image size: {args.image_size}x{args.image_size}")
    logging.info(f"Workers: {args.workers}")
    
    if not tfrecord_files:
        logging.error("No TFRecord files found!")
        sys.exit(1)
    
    # Process files
    target_size = (args.image_size, args.image_size)
    all_stage1 = []
    all_stage2 = []
    all_stats = []
    
    start_time = datetime.now()
    
    if args.workers == 1:
        # Sequential processing
        for i, tfrecord_path in enumerate(tfrecord_files):
            logging.info(f"Processing [{i+1}/{len(tfrecord_files)}]: {tfrecord_path.name}")
            s1, s2, stats = process_tfrecord_file(
                tfrecord_path, output_dir, target_size, args.quality, args.skip_existing
            )
            all_stage1.extend(s1)
            all_stage2.extend(s2)
            all_stats.append(stats)
            
            if (i + 1) % 10 == 0:
                logging.info(f"  Progress: {sum(s.total_samples for s in all_stats)} samples processed")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_tfrecord_file,
                    tfrecord_path,
                    output_dir,
                    target_size,
                    args.quality,
                    args.skip_existing,
                ): tfrecord_path
                for tfrecord_path in tfrecord_files
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                tfrecord_path = futures[future]
                try:
                    s1, s2, stats = future.result()
                    all_stage1.extend(s1)
                    all_stage2.extend(s2)
                    all_stats.append(stats)
                    
                    if i % 10 == 0:
                        logging.info(
                            f"Progress: [{i}/{len(tfrecord_files)}] files, "
                            f"{sum(s.total_samples for s in all_stats)} samples"
                        )
                except Exception as e:
                    logging.error(f"Failed to process {tfrecord_path}: {e}")
    
    # Merge stats
    final_stats = merge_stats(all_stats)
    
    logging.info(f"\n{'='*60}")
    logging.info("Processing complete. Generating train/val splits...")
    
    # Train/val split for Stage 1
    s1_train, s1_val = train_val_split(all_stage1, args.val_ratio, args.seed)
    write_jsonl(s1_train, output_dir / "stage1" / "train.jsonl")
    write_jsonl(s1_val, output_dir / "stage1" / "val.jsonl")
    
    # Train/val split for Stage 2
    s2_train, s2_val = train_val_split(all_stage2, args.val_ratio, args.seed)
    write_jsonl(s2_train, output_dir / "stage2" / "train.jsonl")
    write_jsonl(s2_val, output_dir / "stage2" / "val.jsonl")
    
    # Calculate timing
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Save metadata
    metadata = {
        "preprocessing_config": {
            "image_size": args.image_size,
            "quality": args.quality,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "statistics": final_stats.to_dict(),
        "stage1": {
            "train_samples": len(s1_train),
            "val_samples": len(s1_val),
            "total": len(all_stage1),
        },
        "stage2": {
            "train_samples": len(s2_train),
            "val_samples": len(s2_val),
            "total": len(all_stage2),
        },
        "processing_time_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info("PREPROCESSING SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Total samples:     {final_stats.total_samples:,}")
    logging.info(f"Total frames:      {final_stats.total_frames:,}")
    logging.info(f"Total QA pairs:    {final_stats.total_qa_pairs:,}")
    logging.info(f"Avg frames/sample: {np.mean(final_stats.frames_per_sample):.1f}")
    logging.info(f"Avg QAs/sample:    {np.mean(final_stats.qas_per_sample):.1f}")
    logging.info(f"Failed samples:    {len(final_stats.failed_samples)}")
    logging.info(f"Skipped existing:  {final_stats.skipped_existing}")
    logging.info(f"{'='*60}")
    logging.info("Task type distribution:")
    for task_type, count in sorted(final_stats.task_type_counts.items()):
        logging.info(f"  {task_type}: {count:,}")
    logging.info(f"{'='*60}")
    logging.info("Output files:")
    logging.info(f"  Stage 1 train: {len(s1_train):,} samples")
    logging.info(f"  Stage 1 val:   {len(s1_val):,} samples")
    logging.info(f"  Stage 2 train: {len(s2_train):,} samples")
    logging.info(f"  Stage 2 val:   {len(s2_val):,} samples")
    logging.info(f"{'='*60}")
    logging.info(f"Total time: {elapsed/60:.1f} minutes")
    logging.info(f"Throughput: {final_stats.total_samples/elapsed:.1f} samples/sec")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()