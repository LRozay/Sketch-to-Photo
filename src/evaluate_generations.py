"""Score Check-in 2 generations with CLIP, FID, and a pairwise survey sheet.

CLIP score compares each generated image with its prompt. FID compares a
folder of generated images with a folder of real COCO photographs. The survey
sheet lists each sketch, prompt, baseline photo, and LoRA photo so a rater can
mark which one follows the sketch. This script does not fill in scores by hand.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3
from transformers import CLIPModel, CLIPProcessor

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _embedding(output: torch.Tensor | object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    raise SystemExit("CLIP did not return image or text embeddings.")


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def clip_scores(image_dir: Path, prompt_dir: Path, device: torch.device) -> float:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for image_path in list_images(image_dir):
            prompt_path = prompt_dir / f"{image_path.stem}.txt"
            if not prompt_path.is_file():
                continue
            image = Image.open(image_path).convert("RGB")
            prompt = prompt_path.read_text().strip()
            inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            image_features = _embedding(model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = _embedding(
                model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores.append(float((image_features * text_features).sum().item()))
    if not scores:
        raise SystemExit(f"No image and prompt pairs found in {image_dir} and {prompt_dir}.")
    mean_score = float(np.mean(scores))
    print(f"CLIP score over {len(scores)} images: {mean_score:.4f}")
    return mean_score


def inception_features(directory: Path, device: torch.device) -> torch.Tensor:
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for path in list_images(directory):
            image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            features.append(model(image).squeeze(0).float().cpu())
    if len(features) < 2:
        raise SystemExit(f"FID needs at least two images in {directory}.")
    return torch.stack(features)


def frechet_distance(real: torch.Tensor, generated: torch.Tensor) -> float:
    mu_real = real.mean(dim=0)
    mu_generated = generated.mean(dim=0)
    sigma_real = torch.cov(real.T)
    sigma_generated = torch.cov(generated.T)
    cov_mean = matrix_sqrt(sigma_real @ sigma_generated)
    diff = mu_real - mu_generated
    distance = diff.dot(diff) + torch.trace(sigma_real + sigma_generated - 2 * cov_mean)
    return float(distance.real.item())


def matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp(min=0).sqrt()
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def fid_score(real_dir: Path, generated_dir: Path, device: torch.device) -> float:
    distance = frechet_distance(
        inception_features(real_dir, device),
        inception_features(generated_dir, device),
    )
    print(f"FID: {distance:.4f}")
    return distance


def write_survey(sketch_dir: Path, prompt_dir: Path, baseline_dir: Path, lora_dir: Path, output: Path) -> None:
    rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sketch", "prompt", "baseline_image", "lora_image", "follows_sketch_better"])
        for sketch in list_images(sketch_dir):
            prompt_path = prompt_dir / f"{sketch.stem}.txt"
            baseline = baseline_dir / f"{sketch.stem}.png"
            lora = lora_dir / f"{sketch.stem}.png"
            if not (prompt_path.is_file() and baseline.is_file() and lora.is_file()):
                continue
            writer.writerow([sketch, prompt_path.read_text().strip(), baseline, lora, ""])
            rows += 1
    if rows == 0:
        raise SystemExit("No complete sketch, prompt, baseline, and LoRA sets were found for the survey.")
    print(f"wrote {rows} survey rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Check-in 2 sketch-to-photo generations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clip_parser = subparsers.add_parser("clip", help="Mean CLIP similarity of generated images and prompts")
    clip_parser.add_argument("--images", required=True)
    clip_parser.add_argument("--prompts", required=True)

    fid_parser = subparsers.add_parser("fid", help="FID between real photographs and generated images")
    fid_parser.add_argument("--real", required=True)
    fid_parser.add_argument("--generated", required=True)

    survey_parser = subparsers.add_parser("survey", help="Write the pairwise human-rating sheet")
    survey_parser.add_argument("--sketches", required=True)
    survey_parser.add_argument("--prompts", required=True)
    survey_parser.add_argument("--baseline", required=True)
    survey_parser.add_argument("--lora", required=True)
    survey_parser.add_argument("--output", default="outputs/pairwise_survey.csv")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.command == "clip":
        clip_scores(Path(args.images), Path(args.prompts), device)
    elif args.command == "fid":
        fid_score(Path(args.real), Path(args.generated), device)
    else:
        write_survey(Path(args.sketches), Path(args.prompts), Path(args.baseline), Path(args.lora), Path(args.output))


if __name__ == "__main__":
    main()
