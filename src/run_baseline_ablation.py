"""Sweep conditioning scale and guidance scale for baseline or LoRA adapter.

The project plan compares these two controls on the pretrained canny and
scribble ControlNets before any LoRA fine-tune, then repeats the same
nine-cell grid after the adapter is loaded. Seed stays fixed so cells differ
by scale, not by a new noise draw. Each cell calls baseline_controlnet.py once
and writes its own image.
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
    parser = argparse.ArgumentParser(description="Ablate ControlNet conditioning and guidance scales")
    parser.add_argument("--sketch", required=True, help="Sketch, scribble, or source image for conditioning")
    parser.add_argument("--prompt", required=True, help="Text prompt paired with the sketch")
    parser.add_argument("--control-type", choices=sorted(CONTROLNET_MODEL_IDS), default="scribble")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--apply-canny", action="store_true", help="Convert the input to a Canny edge map first")
    parser.add_argument("--lora-path", default=None, help="Repeat the nine-cell grid with a ControlNet LoRA adapter")
    parser.add_argument("--seed", type=int, default=261)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "baseline_controlnet.py"
    label = "lora" if args.lora_path else "baseline"

    for conditioning_scale in CONDITIONING_SCALES:
        for guidance_scale in GUIDANCE_SCALES:
            name = f"{label}_{args.control_type}_cond{conditioning_scale}_guid{guidance_scale}.png"
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
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--output",
                str(output_dir / name),
            ]
            if args.apply_canny:
                command.append("--apply-canny")
            if args.lora_path:
                command.extend(["--lora-path", args.lora_path])
            subprocess.run(command, check=True)
    print(f"wrote nine-cell {label} grid to {output_dir}")


if __name__ == "__main__":
    main()
