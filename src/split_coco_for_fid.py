"""Hold out COCO photographs for FID against the larger generated set.

FID on only 50 to 100 hand-drawn images is noisy, so Check-in 2 scores
realism on held-out COCO photographs. This script reads the prepared subset
stems and writes a disjoint fid_real split while leaving the remaining stems
for synthetic pairing and generation.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_stems(path: Path) -> list[str]:
    stems = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not stems:
        raise SystemExit(f"No image stems found in {path}.")
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the COCO subset into pairing and FID real sets")
    parser.add_argument("--subset-list", default="data/coco/splits/train.txt", help="Prepared COCO stem list")
    parser.add_argument("--output-dir", default="data/coco/splits", help="Folder for train and fid_real lists")
    parser.add_argument("--fid-count", type=int, default=500, help="Number of held-out real photographs for FID")
    parser.add_argument("--seed", type=int, default=261)
    args = parser.parse_args()

    stems = read_stems(Path(args.subset_list))
    if args.fid_count < 2:
        raise SystemExit("FID needs at least two held-out photographs.")
    if args.fid_count >= len(stems):
        raise SystemExit(f"--fid-count {args.fid_count} leaves no stems for synthetic pairing.")

    rng = random.Random(args.seed)
    fid_stems = sorted(rng.sample(stems, args.fid_count))
    fid_set = set(fid_stems)
    pair_stems = [stem for stem in stems if stem not in fid_set]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("fid_real.txt").write_text("\n".join(fid_stems) + "\n")
    output_dir.joinpath("pair_train.txt").write_text("\n".join(pair_stems) + "\n")
    print(f"wrote {len(fid_stems)} FID real stems and {len(pair_stems)} pairing stems under {output_dir}")


if __name__ == "__main__":
    main()
