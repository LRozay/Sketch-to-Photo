"""Run the pretrained scribble baseline over every complete hand-drawn triple.

The evaluation plan runs the baseline end to end on the same sketches that
later receive the LoRA adapter. Seed, scales, and step count stay fixed so
later pairs differ by the model, not by a new noise draw.
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
        if candidate.exists():
            return candidate
    return None


def complete_triples(data_dir: Path) -> list[tuple[str, Path, str]]:
    sketch_dir = data_dir / "sketches"
    image_dir = data_dir / "images"
    prompt_dir = data_dir / "prompts"
    triples: list[tuple[str, Path, str]] = []
    if not sketch_dir.is_dir():
        return triples
    for sketch_path in sorted(sketch_dir.iterdir()):
        if sketch_path.suffix.lower() not in IMAGE_SUFFIXES or sketch_path.name.startswith("."):
            continue
        if find_with_stem(image_dir, sketch_path.stem) is None:
            continue
        prompt_path = prompt_dir / f"{sketch_path.stem}.txt"
        if not prompt_path.is_file():
            continue
        triples.append((sketch_path.stem, sketch_path, prompt_path.read_text().strip()))
    return triples


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch pretrained ControlNet baseline over hand-drawn triples")
    parser.add_argument("--data-dir", default="data/hand_drawn")
    parser.add_argument("--control-type", choices=sorted(CONTROLNET_MODEL_IDS), default="scribble")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=261)
    parser.add_argument("--lora-path", default=None, help="Optional ControlNet LoRA adapter directory")
    args = parser.parse_args()

    triples = complete_triples(Path(args.data_dir))
    if not triples:
        raise SystemExit(f"No complete triples found in {args.data_dir}. Finish the hand-drawn set first.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "baseline_controlnet.py"

    for stem, sketch_path, prompt in triples:
        command = [
            sys.executable,
            str(script),
            "--sketch",
            str(sketch_path),
            "--prompt",
            prompt,
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
        ]
        if args.lora_path:
            command.extend(["--lora-path", args.lora_path])
        subprocess.run(command, check=True)
    print(f"wrote {len(triples)} {args.control_type} generations to {output_dir}")


if __name__ == "__main__":
    main()
