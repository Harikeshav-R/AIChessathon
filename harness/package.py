"""Packages tournament submission into submission.zip."""

from __future__ import annotations

import argparse
import zipfile
from collections.abc import Iterator
from pathlib import Path

from harness.rules import MAX_UNZIPPED_BYTES

DEFAULT_INCLUDES = ("weights", "engine", "assets")
SKIP = {"__pycache__", ".DS_Store"}


def members(root: Path, includes: tuple[str, ...]) -> Iterator[tuple[Path, str]]:
    seen_relative: set[str] = set()
    for path in sorted(root.glob("*.py")):
        if path.name != "convert_starter_weights.py" and path.name not in seen_relative:
            seen_relative.add(path.name)
            yield path, path.name
    for name in includes:
        source = root / name
        if source.is_file():
            rel = str(source.relative_to(root))
            if rel not in seen_relative:
                seen_relative.add(rel)
                yield source, rel
        elif source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and not SKIP & set(path.parts):
                    rel = str(path.relative_to(root))
                    if rel not in seen_relative:
                        seen_relative.add(rel)
                        yield path, rel


def build(root: Path, destination: Path, includes: tuple[str, ...]) -> list[str]:
    entries = list(members(root, includes))
    written = [name for _, name in entries]
    if "agent.py" not in written:
        raise SystemExit(f"{root / 'agent.py'} does not exist; the platform imports it by name")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name in entries:
            archive.write(source, name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a submission zip.")
    parser.add_argument("--out", type=Path, default=Path("submission.zip"))
    parser.add_argument("--include", action="append", default=[])
    arguments = parser.parse_args()

    includes = DEFAULT_INCLUDES + tuple(arguments.include)
    root = Path.cwd()
    written = build(root, arguments.out, includes)
    size = arguments.out.stat().st_size
    unzipped = sum((root / name).stat().st_size for name in written)
    print(f"{arguments.out} ({size:,} bytes, {unzipped:,} unzipped)")
    for name in written:
        print(f"  {name}")
    if unzipped > MAX_UNZIPPED_BYTES:
        over = unzipped / MAX_UNZIPPED_BYTES
        print(
            f"\nwarning: {unzipped / 1024 / 1024:.1f} MB unzipped is {over:.1f}x the "
            f"{MAX_UNZIPPED_BYTES // 1024 // 1024} MB limit. The platform will reject this upload"
        )


if __name__ == "__main__":
    main()
