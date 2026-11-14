"""Generate the larger COCO set used for FID in Check-in 3 and Check-in 4.

FID is scored against the 500 held-out photographs in data/coco/fid_real.
This script walks the pairing stems, loads each photo's prompt, and runs the
pretrained ControlNet on the matching Canny or HED edge map. Outputs go to
outputs/baseline_coco/ for src/record_baseline_metrics.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_setup import CONTROLNET_MODEL_IDS

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def find_with_stem(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ControlNet generations on COCO edge maps for FID")
    parser.add_argument("--pair-list", default="data/coco/splits/pair_train.txt")
    parser.add_argument("--images", default="data/coco/images")
    parser.add_argument("--edges", default="data/coco/canny", help="Canny or HED maps with matching stems")
    parser.add_argument("--prompts", default="data/coco/prompts")
    parser.add_argument("--control-type", choices=sorted(CONTROLNET_MODEL_IDS), default="canny")
    parser.add_argument("--output-dir", default="outputs/baseline_coco")
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=261)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for a short dry run; 0 means all")
    args = parser.parse_args()

    pair_list = Path(args.pair_list)
    if not pair_list.is_file():
        raise SystemExit(f"Missing {pair_list}. Run src/split_coco_for_fid.py first.")

    stems = [line.strip() for line in pair_list.read_text().splitlines() if line.strip()]
    if args.limit > 0:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"No pairing stems found in {pair_list}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "baseline_controlnet.py"
    images_dir = Path(args.images)
    edges_dir = Path(args.edges)
    prompts_dir = Path(args.prompts)

    written = 0
    skipped = 0
    for stem in stems:
        edge = find_with_stem(edges_dir, stem)
        prompt_path = prompts_dir / f"{stem}.txt"
        if edge is None or not prompt_path.is_file():
            # Fall back to the photo itself only when edges are missing and canny is requested.
            if args.control_type == "canny" and find_with_stem(images_dir, stem) is not None and prompt_path.is_file():
                edge = find_with_stem(images_dir, stem)
                apply_canny = True
            else:
                skipped += 1
                continue
        else:
            apply_canny = False

        command = [
            sys.executable,
            str(script),
            "--sketch",
            str(edge),
            "--prompt",
            prompt_path.read_text().strip(),
            "--control-type",
            args.control_type,
            "--conditioning-scale",
            str(args.conditioning_scale),
            "--guidance-scale",
            str(args.guidance_scale),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--output",
            str(output_dir / f"{stem}.png"),
            "--no-invert-scribble",
        ]
        if apply_canny:
            command.append("--apply-canny")
        subprocess.run(command, check=True)
        written += 1

    if written == 0:
        raise SystemExit(f"No COCO generations written. Skipped {skipped} stems missing edges or prompts.")
    print(f"wrote {written} generations to {output_dir} ({skipped} skipped)")


if __name__ == "__main__":
    main()
