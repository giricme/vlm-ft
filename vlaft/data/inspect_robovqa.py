"""Inspect downloaded RoboVQA dataset.

Comprehensive inspection of downloaded TFRecord files:
- Completeness check (file integrity)
- Frame count statistics per sample
- Image dimensions
- QA field analysis
- Overall dataset statistics

Usage:
    # Full inspection
    python -m vlaft.data.inspect_robovqa --data-dir /data/robovqa/raw
    python -m vlaft.data.inspect_robovqa --data-dir /data/robovqa/raw --max-samples 1000

    # Sample mode: dump N random complete records
    python -m vlaft.data.inspect_robovqa --data-dir /data/robovqa/raw --sample 12
    python -m vlaft.data.inspect_robovqa --data-dir /data/robovqa/raw --sample 12 --seed 123
"""

import argparse
from collections import defaultdict
from datetime import datetime
import glob
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# TensorFlow import with reduced logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


def setup_logging(output_dir: Path) -> None:
    """Set up root logger with console and file handlers."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"inspect_{timestamp}.log"

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


def find_tfrecord_files(data_dir: Path) -> List[Path]:
    """Find all TFRecord files in the data directory."""
    patterns = [
        str(data_dir / "**" / "*.tfrecord"),
        str(data_dir / "**" / "*.tfrecord-*"),
        str(data_dir / "**" / "train*"),
        str(data_dir / "**" / "val*"),
        str(data_dir / "**" / "test*"),
    ]

    files = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))

    # Filter to actual files (not directories) and TFRecord-like files
    tfrecord_files = []
    for f in files:
        path = Path(f)
        if path.is_file() and path.stat().st_size > 0:
            tfrecord_files.append(path)

    return sorted(tfrecord_files)


def check_file_integrity(filepath: Path) -> Tuple[bool, str, int]:
    """
    Check if a TFRecord file is readable and count records.

    Returns:
        Tuple of (is_valid, error_message, record_count)
    """
    try:
        dataset = tf.data.TFRecordDataset(str(filepath))
        count = 0
        for _ in dataset:
            count += 1
        return (True, "", count)
    except tf.errors.DataLossError as e:
        return (False, f"Data corruption: {e}", 0)
    except Exception as e:
        return (False, str(e), 0)


def parse_sequence_example(raw_record: bytes) -> Dict[str, Any]:
    """
    Parse a single TFRecord SequenceExample.

    Returns dict with:
        - num_frames: int
        - image_sizes: list of (height, width, channels)
        - text_fields: dict of field_name -> value
        - feature_names: list of all feature names found
    """
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record)

    result = {
        "num_frames": 0,
        "image_sizes": [],
        "image_bytes_sizes": [],
        "text_fields": {},
        "context_features": [],
        "sequence_features": [],
    }

    # Parse context features (non-sequential)
    for key, feature in example.context.feature.items():
        result["context_features"].append(key)

        # Try to extract text values
        if feature.bytes_list.value:
            try:
                value = feature.bytes_list.value[0].decode("utf-8")
                result["text_fields"][key] = value
            except (UnicodeDecodeError, IndexError):
                pass
        elif feature.int64_list.value:
            result["text_fields"][key] = list(feature.int64_list.value)
        elif feature.float_list.value:
            result["text_fields"][key] = list(feature.float_list.value)

    # Parse sequence features (frames, etc.)
    for key, feature_list in example.feature_lists.feature_list.items():
        result["sequence_features"].append(key)

        if key.lower() in ["images", "image", "frames", "frame", "video"]:
            num_frames = len(feature_list.feature)
            result["num_frames"] = num_frames

            # Sample a few frames to get dimensions
            for i, feature in enumerate(feature_list.feature):
                if feature.bytes_list.value:
                    img_bytes = feature.bytes_list.value[0]
                    result["image_bytes_sizes"].append(len(img_bytes))

                    # Decode first, middle, last frame to check dimensions
                    if i == 0 or i == num_frames // 2 or i == num_frames - 1:
                        try:
                            img = tf.image.decode_image(img_bytes)
                            result["image_sizes"].append(tuple(img.shape.as_list()))
                        except Exception:
                            pass

    return result


def dump_full_record(raw_record: bytes) -> Dict[str, Any]:
    """
    Extract all content from a single TFRecord for detailed inspection.

    Returns dict with all context and sequence features fully decoded.
    """
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record)

    record = {
        "context": {},
        "sequences": {},
    }

    # Parse all context features
    for key, feature in example.context.feature.items():
        if feature.bytes_list.value:
            try:
                value = feature.bytes_list.value[0].decode("utf-8")
                record["context"][key] = value
            except UnicodeDecodeError:
                record["context"][
                    key
                ] = f"<binary: {len(feature.bytes_list.value[0])} bytes>"
        elif feature.int64_list.value:
            record["context"][key] = list(feature.int64_list.value)
        elif feature.float_list.value:
            record["context"][key] = list(feature.float_list.value)

    # Parse all sequence features
    for key, feature_list in example.feature_lists.feature_list.items():
        num_items = len(feature_list.feature)

        if key.lower() in ["images", "image", "frames", "frame", "video"]:
            # For images, just report count and sizes
            sizes = []
            for feature in feature_list.feature:
                if feature.bytes_list.value:
                    sizes.append(len(feature.bytes_list.value[0]))
            record["sequences"][key] = {
                "type": "images",
                "count": num_items,
                "byte_sizes": sizes[:5] if len(sizes) > 5 else sizes,
                "note": f"... and {len(sizes) - 5} more" if len(sizes) > 5 else None,
            }
        else:
            # For text and other sequences, extract all values
            values = []
            for feature in feature_list.feature:
                if feature.bytes_list.value:
                    try:
                        values.append(feature.bytes_list.value[0].decode("utf-8"))
                    except UnicodeDecodeError:
                        values.append(
                            f"<binary: {len(feature.bytes_list.value[0])} bytes>"
                        )
                elif feature.int64_list.value:
                    values.append(list(feature.int64_list.value))
                elif feature.float_list.value:
                    values.append(list(feature.float_list.value))

            record["sequences"][key] = {
                "type": "sequence",
                "count": num_items,
                "values": values,
            }

    return record


def sample_records(
    data_dir: Path,
    num_samples: int = 12,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Randomly sample records from across the dataset.

    Args:
        data_dir: Directory containing TFRecord files
        num_samples: Number of records to sample
        seed: Random seed for reproducibility

    Returns:
        List of fully dumped records
    """
    random.seed(seed)

    tfrecord_files = find_tfrecord_files(data_dir)
    if not tfrecord_files:
        logging.error("No TFRecord files found")
        return []

    logging.info(f"Found {len(tfrecord_files)} TFRecord files")

    # First pass: count records per file to enable random sampling
    logging.info("Counting records per file...")
    file_record_counts = []
    total_records = 0

    for filepath in tfrecord_files:
        try:
            dataset = tf.data.TFRecordDataset(str(filepath))
            count = sum(1 for _ in dataset)
            file_record_counts.append((filepath, count, total_records))
            total_records += count
        except Exception as e:
            logging.warning(f"Error reading {filepath}: {e}")

    logging.info(f"Total records: {total_records}")

    # Generate random indices
    if num_samples >= total_records:
        sample_indices = list(range(total_records))
    else:
        sample_indices = sorted(random.sample(range(total_records), num_samples))

    logging.info(f"Sampling {len(sample_indices)} records...")

    # Map indices to files and fetch records
    sampled_records = []
    current_idx = 0

    for filepath, count, start_idx in file_record_counts:
        end_idx = start_idx + count

        # Find which sample indices fall in this file
        indices_in_file = [
            idx - start_idx for idx in sample_indices if start_idx <= idx < end_idx
        ]

        if not indices_in_file:
            continue

        # Read the specific records from this file
        try:
            dataset = tf.data.TFRecordDataset(str(filepath))
            for local_idx, raw_record in enumerate(dataset):
                if local_idx in indices_in_file:
                    record = dump_full_record(raw_record.numpy())
                    record["_meta"] = {
                        "global_index": start_idx + local_idx,
                        "file": filepath.name,
                        "local_index": local_idx,
                    }
                    sampled_records.append(record)

                    if len(sampled_records) >= num_samples:
                        break

        except Exception as e:
            logging.warning(f"Error sampling from {filepath}: {e}")

        if len(sampled_records) >= num_samples:
            break

    return sampled_records


def print_sampled_records(records: List[Dict[str, Any]]) -> None:
    """Print sampled records in a human-readable format."""
    for i, record in enumerate(records):
        logging.info("")
        logging.info("=" * 70)
        logging.info(f"RECORD {i + 1}/{len(records)}")
        logging.info("=" * 70)

        meta = record.get("_meta", {})
        logging.info(f"  File: {meta.get('file', 'unknown')}")
        logging.info(f"  Global index: {meta.get('global_index', 'unknown')}")

        logging.info("")
        logging.info("  CONTEXT FEATURES:")
        logging.info("  " + "-" * 38)
        for key, value in record.get("context", {}).items():
            logging.info(f"    {key}: {value}")

        logging.info("")
        logging.info("  SEQUENCE FEATURES:")
        logging.info("  " + "-" * 38)
        for key, seq_data in record.get("sequences", {}).items():
            seq_type = seq_data.get("type", "unknown")
            count = seq_data.get("count", 0)

            if seq_type == "images":
                sizes = seq_data.get("byte_sizes", [])
                logging.info(f"    {key}: [{count} images]")
                logging.info(f"      Sample sizes (bytes): {sizes}")
            else:
                values = seq_data.get("values", [])
                logging.info(f"    {key}: [{count} items]")
                for j, v in enumerate(values):
                    if isinstance(v, str):
                        preview = v[:200] + "..." if len(v) > 200 else v
                        logging.info(f"      [{j}]: {preview}")
                    else:
                        logging.info(f"      [{j}]: {v}")

    logging.info("")
    logging.info("=" * 70)


def analyze_sample(raw_record: bytes, sample_idx: int) -> Optional[Dict[str, Any]]:
    """Analyze a single sample and return statistics."""
    try:
        parsed = parse_sequence_example(raw_record)
        return {
            "sample_idx": sample_idx,
            "num_frames": parsed["num_frames"],
            "image_sizes": parsed["image_sizes"],
            "avg_image_bytes": (
                np.mean(parsed["image_bytes_sizes"])
                if parsed["image_bytes_sizes"]
                else 0
            ),
            "text_fields": list(parsed["text_fields"].keys()),
            "text_values": parsed["text_fields"],
            "context_features": parsed["context_features"],
            "sequence_features": parsed["sequence_features"],
        }
    except Exception as e:
        logging.warning(f"Failed to parse sample {sample_idx}: {e}")
        return None


def compute_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics from analyzed samples."""
    if not samples:
        return {}

    frame_counts = [s["num_frames"] for s in samples if s["num_frames"] > 0]
    avg_image_bytes = [
        s["avg_image_bytes"] for s in samples if s["avg_image_bytes"] > 0
    ]

    # Collect all unique text fields
    all_text_fields = set()
    for s in samples:
        all_text_fields.update(s["text_fields"])

    # Collect all unique features
    all_context_features = set()
    all_sequence_features = set()
    for s in samples:
        all_context_features.update(s["context_features"])
        all_sequence_features.update(s["sequence_features"])

    # Image size analysis
    all_image_sizes = []
    for s in samples:
        all_image_sizes.extend(s["image_sizes"])

    unique_sizes = list(set(all_image_sizes))

    # Sample text field values (first few samples)
    sample_texts = {}
    for field in list(all_text_fields)[:10]:
        values = []
        for s in samples[:5]:
            if field in s["text_values"]:
                val = s["text_values"][field]
                if isinstance(val, str) and len(val) < 500:
                    values.append(val)
        if values:
            sample_texts[field] = values

    stats = {
        "total_samples_analyzed": len(samples),
        "samples_with_frames": len(frame_counts),
        "frame_count_stats": {
            "min": int(np.min(frame_counts)) if frame_counts else 0,
            "max": int(np.max(frame_counts)) if frame_counts else 0,
            "mean": float(np.mean(frame_counts)) if frame_counts else 0,
            "median": float(np.median(frame_counts)) if frame_counts else 0,
            "std": float(np.std(frame_counts)) if frame_counts else 0,
            "percentiles": {
                "p10": float(np.percentile(frame_counts, 10)) if frame_counts else 0,
                "p25": float(np.percentile(frame_counts, 25)) if frame_counts else 0,
                "p75": float(np.percentile(frame_counts, 75)) if frame_counts else 0,
                "p90": float(np.percentile(frame_counts, 90)) if frame_counts else 0,
                "p99": float(np.percentile(frame_counts, 99)) if frame_counts else 0,
            },
        },
        "image_stats": {
            "unique_sizes": unique_sizes,
            "avg_bytes_per_image": (
                float(np.mean(avg_image_bytes)) if avg_image_bytes else 0
            ),
        },
        "text_fields": sorted(list(all_text_fields)),
        "context_features": sorted(list(all_context_features)),
        "sequence_features": sorted(list(all_sequence_features)),
        "sample_text_values": sample_texts,
    }

    # Frame count distribution (histogram buckets)
    if frame_counts:
        hist, bin_edges = np.histogram(
            frame_counts,
            bins=[0, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf")],
        )
        stats["frame_count_distribution"] = {
            f"{int(bin_edges[i])}-{int(bin_edges[i+1]) if bin_edges[i+1] != float('inf') else 'inf'}": int(
                hist[i]
            )
            for i in range(len(hist))
        }

    return stats


def inspect_dataset(
    data_dir: Path,
    max_samples: Optional[int] = None,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full inspection of the RoboVQA dataset.

    Args:
        data_dir: Directory containing downloaded TFRecord files
        max_samples: Maximum samples to analyze (None for all)
        max_files: Maximum files to process (None for all)

    Returns:
        Complete inspection report as dictionary
    """
    report = {
        "inspection_time": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "file_summary": {},
        "integrity_check": {},
        "sample_statistics": {},
        "errors": [],
    }

    # Find all TFRecord files
    logging.info(f"Scanning for TFRecord files in {data_dir}...")
    tfrecord_files = find_tfrecord_files(data_dir)
    logging.info(f"Found {len(tfrecord_files)} TFRecord files")

    if not tfrecord_files:
        report["errors"].append("No TFRecord files found")
        return report

    if max_files:
        tfrecord_files = tfrecord_files[:max_files]
        logging.info(f"Limiting to {max_files} files")

    # File summary
    total_size = sum(f.stat().st_size for f in tfrecord_files)
    report["file_summary"] = {
        "total_files": len(tfrecord_files),
        "total_size_gb": total_size / (1024**3),
        "files": [
            {
                "path": str(f.relative_to(data_dir)),
                "size_mb": f.stat().st_size / (1024**2),
            }
            for f in tfrecord_files[:20]  # List first 20 files
        ],
    }
    if len(tfrecord_files) > 20:
        report["file_summary"][
            "note"
        ] = f"... and {len(tfrecord_files) - 20} more files"

    logging.info(f"Total size: {total_size / (1024**3):.2f} GB")

    # Integrity check and sample collection
    logging.info("Checking file integrity and collecting samples...")
    valid_files = 0
    corrupt_files = []
    total_records = 0
    all_samples = []
    sample_count = 0

    for i, filepath in enumerate(tfrecord_files):
        if i % 10 == 0:
            logging.info(
                f"Processing file {i+1}/{len(tfrecord_files)}: {filepath.name}"
            )

        is_valid, error_msg, record_count = check_file_integrity(filepath)

        if is_valid:
            valid_files += 1
            total_records += record_count

            # Analyze samples from this file
            if max_samples is None or sample_count < max_samples:
                try:
                    dataset = tf.data.TFRecordDataset(str(filepath))
                    for raw_record in dataset:
                        if max_samples and sample_count >= max_samples:
                            break

                        sample_data = analyze_sample(raw_record.numpy(), sample_count)
                        if sample_data:
                            all_samples.append(sample_data)

                        sample_count += 1

                        if sample_count % 1000 == 0:
                            logging.info(f"Analyzed {sample_count} samples...")

                except Exception as e:
                    logging.warning(f"Error reading samples from {filepath}: {e}")
        else:
            corrupt_files.append({"file": str(filepath.name), "error": error_msg})

    report["integrity_check"] = {
        "valid_files": valid_files,
        "corrupt_files": len(corrupt_files),
        "total_records": total_records,
        "corrupt_file_details": corrupt_files[:10],  # First 10 corrupt files
    }

    logging.info(f"Integrity check: {valid_files}/{len(tfrecord_files)} files valid")
    logging.info(f"Total records: {total_records}")

    if corrupt_files:
        logging.warning(f"Found {len(corrupt_files)} corrupt files")

    # Compute statistics
    logging.info(f"Computing statistics from {len(all_samples)} samples...")
    report["sample_statistics"] = compute_statistics(all_samples)

    return report


def print_report(report: Dict[str, Any]) -> None:
    """Print a human-readable summary of the inspection report."""
    logging.info("")
    logging.info("=" * 70)
    logging.info("ROBOVQA INSPECTION REPORT")
    logging.info("=" * 70)

    # File summary
    fs = report.get("file_summary", {})
    logging.info("")
    logging.info("FILE SUMMARY")
    logging.info("-" * 40)
    logging.info(f"  Total files:      {fs.get('total_files', 0)}")
    logging.info(f"  Total size:       {fs.get('total_size_gb', 0):.2f} GB")

    # Integrity
    ic = report.get("integrity_check", {})
    logging.info("")
    logging.info("INTEGRITY CHECK")
    logging.info("-" * 40)
    logging.info(f"  Valid files:      {ic.get('valid_files', 0)}")
    logging.info(f"  Corrupt files:    {ic.get('corrupt_files', 0)}")
    logging.info(f"  Total records:    {ic.get('total_records', 0)}")

    # Sample statistics
    ss = report.get("sample_statistics", {})
    logging.info("")
    logging.info("SAMPLE STATISTICS")
    logging.info("-" * 40)
    logging.info(f"  Samples analyzed: {ss.get('total_samples_analyzed', 0)}")

    fcs = ss.get("frame_count_stats", {})
    if fcs:
        logging.info("")
        logging.info("  Frame counts per sample:")
        logging.info(f"    Min:            {fcs.get('min', 0)}")
        logging.info(f"    Max:            {fcs.get('max', 0)}")
        logging.info(f"    Mean:           {fcs.get('mean', 0):.1f}")
        logging.info(f"    Median:         {fcs.get('median', 0):.1f}")
        logging.info(f"    Std:            {fcs.get('std', 0):.1f}")

        percs = fcs.get("percentiles", {})
        if percs:
            logging.info(f"    10th percentile: {percs.get('p10', 0):.1f}")
            logging.info(f"    90th percentile: {percs.get('p90', 0):.1f}")
            logging.info(f"    99th percentile: {percs.get('p99', 0):.1f}")

    fcd = ss.get("frame_count_distribution", {})
    if fcd:
        logging.info("")
        logging.info("  Frame count distribution:")
        for bucket, count in fcd.items():
            logging.info(f"    {bucket}: {count}")

    imgs = ss.get("image_stats", {})
    if imgs:
        logging.info("")
        logging.info("  Image statistics:")
        logging.info(f"    Unique sizes:   {imgs.get('unique_sizes', [])}")
        logging.info(f"    Avg bytes/img:  {imgs.get('avg_bytes_per_image', 0):.0f}")

    # Features found
    logging.info("")
    logging.info("  Context features:")
    for f in ss.get("context_features", []):
        logging.info(f"    - {f}")

    logging.info("")
    logging.info("  Sequence features:")
    for f in ss.get("sequence_features", []):
        logging.info(f"    - {f}")

    # Sample text values
    stv = ss.get("sample_text_values", {})
    if stv:
        logging.info("")
        logging.info("  Sample text field values:")
        for field, values in list(stv.items())[:5]:
            logging.info(f"    {field}:")
            for v in values[:2]:
                preview = v[:100] + "..." if len(v) > 100 else v
                logging.info(f"      - {preview}")

    logging.info("")
    logging.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inspect downloaded RoboVQA dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing downloaded TFRecord files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to analyze for statistics (default: all)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum files to process (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for full report (default: {data-dir}/inspection_report.json)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample mode: randomly dump N complete records with all fields",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    setup_logging(data_dir)

    # Sample mode: just dump N random records
    if args.sample:
        logging.info(f"Sample mode: dumping {args.sample} random records")
        logging.info(f"Random seed: {args.seed}")

        start_time = time.time()
        records = sample_records(data_dir, num_samples=args.sample, seed=args.seed)
        elapsed = time.time() - start_time

        print_sampled_records(records)

        # Save to JSON
        output_path = args.output or str(data_dir / "sampled_records.json")
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logging.info(f"Sampled records saved to: {output_path}")
        logging.info(f"Sampling completed in {elapsed:.1f} seconds")
        return

    # Full inspection mode
    logging.info(f"Inspecting RoboVQA dataset at: {data_dir}")
    if args.max_samples:
        logging.info(f"Limiting analysis to {args.max_samples} samples")

    start_time = time.time()
    report = inspect_dataset(
        data_dir,
        max_samples=args.max_samples,
        max_files=args.max_files,
    )
    elapsed = time.time() - start_time

    report["elapsed_seconds"] = elapsed

    # Print human-readable report
    print_report(report)

    # Save full report to JSON
    output_path = args.output or str(data_dir / "inspection_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logging.info(f"Full report saved to: {output_path}")

    logging.info(f"Inspection completed in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
