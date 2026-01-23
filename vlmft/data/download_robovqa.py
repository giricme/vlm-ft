"""Download RoboVQA dataset from Google Cloud Storage.

Standalone script to download the RoboVQA dataset with parallel downloads
and progress tracking.

Usage:
    python -m vlmft.data.download_robovqa --output /data/robovqa/raw
    python -m vlmft.data.download_robovqa --output /data/robovqa/raw --workers 16
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from pathlib import Path
import sys
from threading import Lock
import time
from typing import List, Tuple

from google.cloud import storage

# Constants
BUCKET_NAME = "anon_robovqa"
PREFIX = "tfrecord/"


def setup_logging(output_dir: Path) -> None:
    """Set up root logger with console and file handlers."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging to {log_file}")


def list_blobs(bucket_name: str, prefix: str) -> List[storage.Blob]:
    """List all blobs in the bucket with the given prefix."""
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return blobs


def download_blob(
    blob: storage.Blob,
    output_dir: Path,
    bucket_name: str,
) -> Tuple[str, int, bool, str]:
    """
    Download a single blob to the output directory.

    Returns:
        Tuple of (blob_name, bytes_downloaded, success, error_message)
    """
    # Preserve directory structure under prefix
    relative_path = blob.name
    local_path = output_dir / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if file exists with matching size
    if local_path.exists() and local_path.stat().st_size == blob.size:
        return (blob.name, 0, True, "skipped")

    try:
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        blob_ref = bucket.blob(blob.name)
        blob_ref.download_to_filename(str(local_path))
        return (blob.name, blob.size, True, "")
    except Exception as e:
        return (blob.name, 0, False, str(e))


class ProgressTracker:
    """Thread-safe progress tracker."""

    def __init__(self, total_files: int, total_bytes: int):
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.completed_files = 0
        self.downloaded_bytes = 0
        self.skipped_files = 0
        self.failed_files = 0
        self.failures: List[Tuple[str, str]] = []
        self.lock = Lock()
        self.start_time = time.time()

    def update(self, blob_name: str, bytes_downloaded: int, success: bool, msg: str):
        with self.lock:
            self.completed_files += 1
            if success:
                if msg == "skipped":
                    self.skipped_files += 1
                else:
                    self.downloaded_bytes += bytes_downloaded
            else:
                self.failed_files += 1
                self.failures.append((blob_name, msg))

    def log_progress(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            throughput = (
                self.downloaded_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0
            )
            pct = (
                (self.completed_files / self.total_files) * 100
                if self.total_files > 0
                else 0
            )

            logging.info(
                f"Progress: {self.completed_files}/{self.total_files} files ({pct:.1f}%) | "
                f"Downloaded: {self.downloaded_bytes / (1024**3):.2f} GB | "
                f"Skipped: {self.skipped_files} | "
                f"Failed: {self.failed_files} | "
                f"Throughput: {throughput:.2f} MB/s"
            )

    def log_summary(self):
        elapsed = time.time() - self.start_time
        logging.info("=" * 60)
        logging.info("Download Summary")
        logging.info("=" * 60)
        logging.info(f"Total files: {self.total_files}")
        logging.info(
            f"Downloaded: {self.completed_files - self.skipped_files - self.failed_files}"
        )
        logging.info(f"Skipped (already exist): {self.skipped_files}")
        logging.info(f"Failed: {self.failed_files}")
        logging.info(f"Total downloaded: {self.downloaded_bytes / (1024**3):.2f} GB")
        logging.info(f"Elapsed time: {elapsed / 3600:.2f} hours")

        if self.failures:
            logging.warning("Failed downloads:")
            for name, err in self.failures[:10]:
                logging.warning(f"  {name}: {err}")
            if len(self.failures) > 10:
                logging.warning(f"  ... and {len(self.failures) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Download RoboVQA dataset from Google Cloud Storage"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log progress every N files (default: 50)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Workers: {args.workers}")
    logging.info(f"Bucket: gs://{BUCKET_NAME}/{PREFIX}")

    # List all blobs
    logging.info("Listing files in bucket...")
    blobs = list_blobs(BUCKET_NAME, PREFIX)

    # Filter to actual files (not directories)
    blobs = [b for b in blobs if b.size > 0]
    total_bytes = sum(b.size for b in blobs)

    logging.info(f"Found {len(blobs)} files ({total_bytes / (1024**3):.2f} GB)")

    if not blobs:
        logging.warning("No files found. Exiting.")
        return

    # Initialize progress tracker
    tracker = ProgressTracker(len(blobs), total_bytes)

    # Download with thread pool
    logging.info("Starting download...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_blob, blob, output_dir, BUCKET_NAME): blob
            for blob in blobs
        }

        for i, future in enumerate(as_completed(futures), 1):
            blob_name, bytes_downloaded, success, msg = future.result()
            tracker.update(blob_name, bytes_downloaded, success, msg)

            if i % args.log_interval == 0:
                tracker.log_progress()

    tracker.log_summary()
    logging.info("Download complete.")


if __name__ == "__main__":
    main()
