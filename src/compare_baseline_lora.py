"""Render the same sketches with the pretrained baseline and the LoRA adapter.

Seed, scales, and step count stay fixed so a visual pair differs by the model,
not by a new noise draw. Outputs land in separate folders for the survey and
for side-by-side grids in the results section.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_setup import CONTROLNET_MODEL_IDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-seed baseline versus LoRA comparison")
    parser.add_argument("--data-dir", default="data/hand_drawn")
    parser.add_argument("--control-type", choices=sorted(CONTROLNET_MODEL_IDS), default="scribble")
    parser.add_argument("--lora-path", required=True, help="Directory of the trained ControlNet LoRA adapter")
    parser.add_argument("--baseline-dir", default="outputs/compare/baseline")
    parser.add_argument("--lora-dir", default="outputs/compare/lora")
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=261)
    args = parser.parse_args()

    if not Path(args.lora_path).is_dir():
        raise SystemExit(f"LoRA adapter not found: {args.lora_path}. Start the first LoRA run before comparing.")

    script = Path(__file__).resolve().parent / "run_batch_baseline.py"
    shared = [
        sys.executable,
        str(script),
        "--data-dir",
        args.data_dir,
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
    ]
    subprocess.run(shared + ["--output-dir", args.baseline_dir], check=True)
    subprocess.run(shared + ["--output-dir", args.lora_dir, "--lora-path", args.lora_path], check=True)
    print(f"wrote fixed-seed pairs under {args.baseline_dir} and {args.lora_dir}")


if __name__ == "__main__":
    main()
