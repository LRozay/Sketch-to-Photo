"""Report which hand-drawn sketches have a matching photo and prompt.

Check-in 1 training only keeps a sketch when images/ and prompts/ contain the
same filename stem. Run this before a LoRA job so missing pairs are visible
without loading Stable Diffusion.
"""

import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def find_with_stem(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the check-in 1 hand-drawn set")
    parser.add_argument("--data-dir", default="data/hand_drawn", help="Folder with images/, sketches/, and prompts/")
    args = parser.parse_args()

    root = Path(args.data_dir)
    sketch_dir = root / "sketches"
    image_dir = root / "images"
    prompt_dir = root / "prompts"
    if not sketch_dir.is_dir():
        raise SystemExit(f"Missing sketches folder: {sketch_dir}")

    complete = 0
    missing = 0
    for sketch_path in sorted(sketch_dir.iterdir()):
        if sketch_path.name.startswith(".") or sketch_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        problems = []
        if find_with_stem(image_dir, sketch_path.stem) is None:
            problems.append("photo")
        if not (prompt_dir / f"{sketch_path.stem}.txt").is_file():
            problems.append("prompt")
        if problems:
            missing += 1
            print(f"missing {', '.join(problems)}: {sketch_path.name}")
        else:
            complete += 1

    print(f"{complete} complete triples, {missing} sketches missing a photo or prompt")
    if complete == 0:
        raise SystemExit("No complete sketch, photo, and prompt triples found.")


if __name__ == "__main__":
    main()
