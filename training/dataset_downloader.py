"""Hugging Face Lichess Position Evaluations Streaming Downloader.

Downloads the 394M / 957M Lichess evaluations dataset in parallel chunks from:
https://huggingface.co/datasets/Lichess/chess-position-evaluations
"""

from __future__ import annotations

import argparse
import concurrent.futures
import urllib.request
from pathlib import Path

HF_BASE_URL = "https://huggingface.co/datasets/Lichess/chess-position-evaluations/tree/main/data"
TOTAL_FILES = 200  # Partitions in repository


def download_chunk(file_idx: int, output_dir: Path) -> Path | None:
    filename = f"train-{file_idx:05d}-of-{TOTAL_FILES:05d}.parquet"
    target_path = output_dir / filename
    if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
        print(f"[{file_idx}/{TOTAL_FILES}] Already downloaded: {filename}")
        return target_path

    url = f"{HF_BASE_URL}/{filename}"
    print(f"[{file_idx}/{TOTAL_FILES}] Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        mb = target_path.stat().st_size / 1e6
        print(f"[{file_idx}/{TOTAL_FILES}] Downloaded {filename} ({mb:.1f} MB)")
        return target_path
    except Exception as e:
        print(f"[{file_idx}/{TOTAL_FILES}] Failed {filename}: {e}")
        if target_path.exists():
            target_path.unlink()
        return None


def download_dataset(output_dir: Path, max_files: int = 200, workers: int = 4) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_chunk, i, output_dir)
            for i in range(min(max_files, TOTAL_FILES))
        ]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                downloaded.append(res)
    print(f"Downloaded {len(downloaded)} parquet partitions.")
    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Lichess evaluated positions")
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    download_dataset(Path(args.output_dir), max_files=args.max_files, workers=args.workers)
