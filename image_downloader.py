"""
Fakeddit image downloader — resumable, disk-budget-aware, concurrent.

Usage:
    # Download 5,000 images from the train split
    python image_downloader.py --split train --output_dir "G:/My Drive/fakeddit/images" --max_images 5000

    # Download up to 50 GB worth of images
    python image_downloader.py --split train --output_dir "G:/My Drive/fakeddit/images" --max_gb 50

    # Use more workers to go faster (default: 16)
    python image_downloader.py --split train --output_dir images --max_images 5000 --workers 32

Safe to Ctrl+C and re-run — already-downloaded images are skipped via
a manifest file (downloaded_ids.txt) written alongside the images.
"""

import argparse
import os
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SPLIT_FILES = {
    "train": "data/multimodal_train.tsv",
    "validate": "data/multimodal_validate.tsv",
    "test": "data/multimodal_test_public.tsv",
}

TIMEOUT_SECONDS = 10
RETRY_ATTEMPTS = 2


def get_dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def download_image(url: str, dest: Path) -> bool:
    """Download a single image. Returns True on success."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(resp, f)
            return True
        except Exception:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="Fakeddit image downloader")
    parser.add_argument(
        "--split",
        choices=["train", "validate", "test"],
        required=True,
        help="Which split to download images for",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images",
        help=(
            "Directory to save images. "
            "Mac: '/Users/you/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/fakeddit/images' "
            "Windows: 'G:/My Drive/fakeddit/images'"
        ),
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Stop after downloading this many images (e.g. --max_images 5000)",
    )
    parser.add_argument(
        "--max_gb",
        type=float,
        default=50.0,
        help="Hard stop when output_dir reaches this size in GB (default: 50)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel download threads (default: 16)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the TSV files (default: data/)",
    )
    args = parser.parse_args()

    tsv_filename = os.path.basename(SPLIT_FILES[args.split])
    tsv_path = Path(args.data_dir) / tsv_filename

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"TSV not found at {tsv_path}. "
            f"Make sure your data/ dir contains the Fakeddit TSV files."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest of already-downloaded IDs so we can skip them
    manifest_path = output_dir / "downloaded_ids.txt"
    if manifest_path.exists():
        with open(manifest_path) as f:
            already_done = set(line.strip() for line in f if line.strip())
    else:
        already_done = set()

    print(f"Loading {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.replace(np.nan, "", regex=True)
    df = df[df["hasImage"] == True]
    df = df[df["image_url"] != ""]
    df = df[~df["id"].astype(str).isin(already_done)]

    if args.max_images is not None:
        df = df.head(args.max_images)

    print(f"  {len(already_done):,} already downloaded, {len(df):,} queued for this run")

    # Thread-safe counters and manifest writer
    lock = threading.Lock()
    failed_count = 0
    success_count = 0
    stop_flag = threading.Event()
    manifest_file = open(manifest_path, "a")

    def worker(row):
        nonlocal failed_count, success_count

        if stop_flag.is_set():
            return

        img_id = str(row["id"])
        dest = output_dir / f"{img_id}.jpg"

        if dest.exists():
            with lock:
                manifest_file.write(img_id + "\n")
                manifest_file.flush()
            return

        # Check disk budget before downloading
        if get_dir_size_gb(output_dir) >= args.max_gb:
            stop_flag.set()
            return

        success = download_image(row["image_url"], dest)
        with lock:
            if success:
                success_count += 1
                manifest_file.write(img_id + "\n")
                manifest_file.flush()
            else:
                failed_count += 1

    rows = [row for _, row in df.iterrows()]

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker, row): row for row in rows}
            with tqdm(total=len(rows), unit="img") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)
                    with lock:
                        pbar.set_postfix(
                            ok=success_count,
                            fail=failed_count,
                            gb=f"{get_dir_size_gb(output_dir):.2f}",
                        )
                    if stop_flag.is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    finally:
        manifest_file.close()

    if stop_flag.is_set():
        print(f"\nDisk budget reached ({args.max_gb} GB). Stopping.")

    total_on_disk = len(list(output_dir.glob("*.jpg")))
    print(f"\nDone.")
    print(f"  Images on disk: {total_on_disk:,}")
    print(f"  Downloaded this run: {success_count:,}")
    print(f"  Failed this run: {failed_count:,}")
    print(f"  Disk used: {get_dir_size_gb(output_dir):.2f} GB")


if __name__ == "__main__":
    main()
