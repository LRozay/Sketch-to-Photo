"""Sweep conditioning scale and guidance scale for the check-in 1 baseline.

The project plan compares these two controls on the pretrained canny and
scribble ControlNets before any LoRA fine-tune. Each cell calls
baseline_controlnet.py once and writes its own image.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_setup import CONTROLNET_MODEL_IDS

CONDITIONING_SCALES = (0.5, 1.0, 1.5)
GUIDANCE_SCALES = (5.0, 7.5, 12.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate the check-in 1 ControlNet baseline")
    parser.add_argument("--sketch", required=True, help="Sketch, scribble, or source image for conditioning")
    parser.add_argument("--prompt", required=True, help="Text prompt paired with the sketch")
    parser.add_argument("--control-type", choices=sorted(CONTROLNET_MODEL_IDS), default="scribble")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--apply-canny", action="store_true", help="Convert the input to a Canny edge map first")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "baseline_controlnet.py"

    for conditioning_scale in CONDITIONING_SCALES:
        for guidance_scale in GUIDANCE_SCALES:
            name = f"{args.control_type}_cond{conditioning_scale}_guid{guidance_scale}.png"
            command = [
                sys.executable,
                str(script),
                "--sketch",
                args.sketch,
                "--prompt",
                args.prompt,
                "--control-type",
                args.control_type,
                "--conditioning-scale",
                str(conditioning_scale),
                "--guidance-scale",
                str(guidance_scale),
                "--output",
                str(output_dir / name),
            ]
            if args.apply_canny:
                command.append("--apply-canny")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
