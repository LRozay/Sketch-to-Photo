"""Select a COCO 2017 photo subset and export captions as prompts.

Check-in 2 uses a 2,000 to 5,000 image subset. Each selected photograph is
resized to 512 x 512, and its first English caption becomes the prompt file
used later for synthetic Canny and HED pairing.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_setup import IMAGE_SIZE


def load_coco_index(annotations_path: Path) -> tuple[dict[int, dict], dict[int, list[str]]]:
    payload = json.loads(annotations_path.read_text())
    images = {item["id"]: item for item in payload["images"]}
    captions: dict[int, list[str]] = defaultdict(list)
    for annotation in payload["annotations"]:
        caption = str(annotation.get("caption", "")).strip()
        if caption:
            captions[annotation["image_id"]].append(caption)
    return images, captions


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a COCO 2017 subset for synthetic sketch pairs")
    parser.add_argument("--images", required=True, help="Folder of COCO train2017 photographs")
    parser.add_argument("--annotations", required=True, help="instances or captions JSON, e.g. captions_train2017.json")
    parser.add_argument("--output", default="data/coco", help="Output root with images/ and prompts/")
    parser.add_argument("--count", type=int, default=3000, help="Target subset size between 2000 and 5000")
    parser.add_argument("--seed", type=int, default=261)
    parser.add_argument("--resolution", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    if not 2000 <= args.count <= 5000:
        raise SystemExit("--count must stay in the proposal range of 2000 to 5000.")

    images, captions = load_coco_index(Path(args.annotations))
    eligible = [
        image_id
        for image_id, info in images.items()
        if captions.get(image_id) and (Path(args.images) / info["file_name"]).is_file()
    ]
    if len(eligible) < args.count:
        raise SystemExit(f"Only {len(eligible)} captioned photographs were found; need {args.count}.")

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(eligible, args.count), key=lambda image_id: images[image_id]["file_name"])

    output = Path(args.output)
    image_dir = output / "images"
    prompt_dir = output / "prompts"
    splits_dir = output / "splits"
    image_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    stems: list[str] = []
    for image_id in selected:
        info = images[image_id]
        source = Path(args.images) / info["file_name"]
        stem = Path(info["file_name"]).stem
        image = Image.open(source).convert("RGB").resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
        image.save(image_dir / f"{stem}.png")
        prompt_dir.joinpath(f"{stem}.txt").write_text(captions[image_id][0] + "\n")
        stems.append(stem)

    splits_dir.joinpath("train.txt").write_text("\n".join(stems) + "\n")
    print(f"wrote {len(stems)} resized photographs and prompts under {output}")


if __name__ == "__main__":
    main()
