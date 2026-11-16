"""Record CLIP and FID for each Check-in 4 experiment configuration.

The three configurations are pretrained canny, pretrained scribble, and
scribble plus the rank-8 LoRA adapter. This helper updates
results/experiment_matrix.json only when real output folders exist. Missing
folders leave the scores null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_generations import clip_scores, fid_score


def maybe_clip(images: Path, prompts: Path, device: torch.device) -> float | None:
    if not images.is_dir() or not prompts.is_dir():
        return None
    pngs = list(images.glob("*.png")) + list(images.glob("*.jpg"))
    if len(pngs) < 1:
        return None
    return clip_scores(images, prompts, device)


def maybe_fid(real: Path, generated: Path, device: torch.device) -> float | None:
    if not real.is_dir() or not generated.is_dir():
        return None
    if len(list(real.glob("*"))) < 2 or len(list(generated.glob("*.png"))) < 2:
        return None
    return fid_score(real, generated, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill CLIP and FID fields in the experiment matrix")
    parser.add_argument("--matrix", default="results/experiment_matrix.json")
    parser.add_argument("--prompts", default="data/hand_drawn/prompts")
    parser.add_argument("--fid-real", default="data/coco/fid_real")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    matrix = json.loads(matrix_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for config in matrix["configurations"]:
        outputs = config.get("outputs", {})
        image_dir = Path(outputs.get("compare") or outputs.get("batch") or outputs.get("single") or "")
        clip_value = maybe_clip(image_dir, Path(args.prompts), device)
        if clip_value is not None:
            config["clip_score"] = clip_value

        coco_dir = outputs.get("coco_fid")
        if coco_dir:
            fid_value = maybe_fid(Path(args.fid_real), Path(coco_dir), device)
            if fid_value is not None:
                config["fid"] = fid_value

        if config.get("clip_score") is not None or config.get("fid") is not None:
            if config.get("status", "").endswith("ready") or "pending" in str(config.get("status", "")):
                config["status"] = "scores_partially_recorded"

    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"updated {matrix_path}")
    for config in matrix["configurations"]:
        print(
            f"{config['name']}: clip={config.get('clip_score')} fid={config.get('fid')} status={config.get('status')}"
        )


if __name__ == "__main__":
    main()
