"""Record baseline FID and CLIP score after the HPC generation jobs finish.

The Check-in 2 report leaves the results section empty until real numbers
exist. This helper runs the evaluation commands and writes
results/baseline_metrics.json without inventing values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_generations import clip_scores, fid_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Record baseline FID and CLIP score into results/")
    parser.add_argument("--images", default="outputs/baseline", help="Generated baseline photos for CLIP")
    parser.add_argument("--prompts", default="data/hand_drawn/prompts")
    parser.add_argument("--real", default="data/coco/images", help="Held-out or COCO real photographs for FID")
    parser.add_argument("--generated-fid", default="outputs/baseline_coco", help="Larger generated set for FID")
    parser.add_argument("--output", default="results/baseline_metrics.json")
    parser.add_argument("--skip-fid", action="store_true", help="Record CLIP only when the COCO set is not ready")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_mean = clip_scores(Path(args.images), Path(args.prompts), device)
    image_count = len(list(Path(args.images).glob("*.png"))) + len(list(Path(args.images).glob("*.jpg")))

    fid_value = None
    if not args.skip_fid:
        fid_value = fid_score(Path(args.real), Path(args.generated_fid), device)

    payload = {
        "status": "recorded",
        "control_type": "scribble",
        "seed": 261,
        "conditioning_scale": 1.0,
        "guidance_scale": 7.5,
        "steps": 30,
        "clip_score": clip_mean,
        "clip_image_count": image_count,
        "fid": fid_value,
        "fid_real_dir": args.real,
        "fid_generated_dir": None if args.skip_fid else args.generated_fid,
        "notes": "Baseline metrics written from evaluate_generations helpers.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
