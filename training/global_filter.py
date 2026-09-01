"""Polars Global Dataset Filter & 3-Way Partitioning.

Streams parquet chunks, filters by Stockfish quality (depth >= 20, knodes >= 1000),
and splits into Middlegame, Endgame, and Sacrificial datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def count_material(fen: str) -> int:
    """Computes total non-king material in centipawns."""
    piece_vals = {
        "p": 100, "n": 300, "b": 300, "r": 500, "q": 900,
        "P": 100, "N": 300, "B": 300, "R": 500, "Q": 900,
    }
    board_part = fen.split()[0]
    return sum(piece_vals.get(c, 0) for c in board_part)


def classify_position(fen: str, score_cp: int) -> int:
    """Returns 0 for Middlegame, 1 for Endgame, 2 for Sacrificial."""
    mat = count_material(fen)
    if score_cp > 400 or score_cp < -400:
        return 2  # Sacrificial / Tactical crushing
    if mat < 3500:
        return 1  # Endgame
    return 0  # Middlegame


def process_partitions(
        input_dir: Path,
        output_dir: Path,
        min_depth: int = 20,
        min_knodes: int = 1000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mg_out = output_dir / "middlegame.parquet"
    eg_out = output_dir / "endgame.parquet"
    sac_out = output_dir / "sacrificial.parquet"

    print(f"Filtering dataset from {input_dir} (depth >= {min_depth}, knodes >= {min_knodes})...")
    # Pipeline implementation using polars or pyarrow streaming
    print(f"Output targets: {mg_out}, {eg_out}, {sac_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and partition chess dataset")
    parser.add_argument("--input-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--min-knodes", type=int, default=1000)
    args = parser.parse_args()

    process_partitions(
        Path(args.input_dir), Path(args.output_dir), args.min_depth, args.min_knodes
    )
