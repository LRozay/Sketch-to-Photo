"""Copy the held-out COCO photographs listed in fid_real.txt into one folder.

Check-in 3 Section 8.4 says FID must use the 500 held-out photos from the
split, not the whole data/coco/images subset. This helper builds that real
folder after split_coco_for_fid.py has written data/coco/splits/fid_real.txt.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the held-out FID real photograph folder")
    parser.add_argument("--fid-list", default="data/coco/splits/fid_real.txt")
    parser.add_argument("--images", default="data/coco/images", help="Prepared COCO subset photographs")
    parser.add_argument("--output", default="data/coco/fid_real", help="Folder of held-out real photos for FID")
    args = parser.parse_args()

    fid_list = Path(args.fid_list)
    if not fid_list.is_file():
        raise SystemExit(f"Missing {fid_list}. Run src/split_coco_for_fid.py first.")

    images_dir = Path(args.images)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for stem in [line.strip() for line in fid_list.read_text().splitlines() if line.strip()]:
        source = find_image(images_dir, stem)
        if source is None:
            missing += 1
            print(f"missing photo for stem: {stem}")
            continue
        shutil.copy2(source, output / source.name)
        copied += 1

    if copied < 2:
        raise SystemExit(f"FID needs at least two held-out photos. Copied {copied}, missing {missing}.")
    print(f"wrote {copied} held-out photographs to {output} ({missing} missing)")


if __name__ == "__main__":
    main()
