"""Summarize how far the 50 to 100 hand-drawn triples have progressed.

The evaluation plan waits on complete sketch, photo, and prompt triples in
data/hand_drawn. This report counts planned rows from the collection
manifest, complete triples on disk, and remaining gaps before the HPC jobs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def find_with_stem(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def list_sketch_stems(sketch_dir: Path) -> list[str]:
    if not sketch_dir.is_dir():
        return []
    return sorted(
        path.stem
        for path in sketch_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report hand-drawn collection progress for Check-in 2 remaining work")
    parser.add_argument("--data-dir", default="data/hand_drawn")
    parser.add_argument("--manifest", default="data/hand_drawn/collection_manifest.csv")
    parser.add_argument("--target-min", type=int, default=50)
    parser.add_argument("--target-max", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.data_dir)
    sketch_dir = root / "sketches"
    image_dir = root / "images"
    prompt_dir = root / "prompts"

    planned = 0
    manifest_path = Path(args.manifest)
    if manifest_path.is_file():
        with manifest_path.open(newline="") as handle:
            planned = sum(1 for _ in csv.DictReader(handle))

    complete = 0
    incomplete: list[str] = []
    for stem in list_sketch_stems(sketch_dir):
        missing = []
        if find_with_stem(image_dir, stem) is None:
            missing.append("photo")
        if not (prompt_dir / f"{stem}.txt").is_file():
            missing.append("prompt")
        if missing:
            incomplete.append(f"{stem} missing {', '.join(missing)}")
        else:
            complete += 1

    sketch_only = len(list_sketch_stems(sketch_dir))
    remaining_to_min = max(args.target_min - complete, 0)
    print(f"manifest planned rows: {planned}")
    print(f"sketches on disk: {sketch_only}")
    print(f"complete triples: {complete}")
    print(f"incomplete sketches: {len(incomplete)}")
    print(f"remaining to reach {args.target_min}-{args.target_max}: {remaining_to_min} more complete triples")
    for line in incomplete[:20]:
        print(line)
    if complete < args.target_min:
        print("Hand-drawn set is not finished yet; keep collecting before the full HPC comparison.")


if __name__ == "__main__":
    main()
