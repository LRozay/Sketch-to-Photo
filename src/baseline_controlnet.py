"""Generate a photo from a sketch with the pretrained ControlNet baseline.

Check-in 1 compares this baseline with a LoRA fine-tune of the same ControlNet.
Canny and scribble checkpoints are both supported. Conditioning scale and guidance
scale are arguments because the project ablates both.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_setup import CONTROLNET_MODEL_IDS, IMAGE_SIZE, SD15_MODEL_ID


def load_rgb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def canny_edges(image: Image.Image, low_threshold: int, high_threshold: int) -> Image.Image:
    """Build the three-channel edge map expected by the canny ControlNet."""
    edges = cv2.Canny(np.array(image), low_threshold, high_threshold)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained Stable Diffusion 1.5 ControlNet baseline")
    parser.add_argument("--sketch", required=True, help="Sketch, scribble, or source image for conditioning")
    parser.add_argument("--prompt", required=True, help="Text prompt paired with the sketch")
    parser.add_argument("--control-type", choices=sorted(CONTROLNET_MODEL_IDS), default="scribble")
    parser.add_argument("--output", default="outputs/baseline.png")
    parser.add_argument("--resolution", type=int, default=IMAGE_SIZE)
    parser.add_argument("--conditioning-scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=261)
    parser.add_argument("--apply-canny", action="store_true", help="Convert the input to a Canny edge map first")
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--lora-path", default=None, help="Directory of a ControlNet LoRA adapter to load")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_IDS[args.control_type], torch_dtype=dtype)
    if args.lora_path:
        controlnet.load_lora_adapter(args.lora_path)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD15_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    condition = load_rgb(Path(args.sketch), args.resolution)
    if args.apply_canny:
        condition = canny_edges(condition, args.canny_low, args.canny_high)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipe(
        args.prompt,
        image=condition,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.conditioning_scale,
        generator=generator,
    ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
