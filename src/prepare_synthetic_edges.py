"""Build Canny or HED conditioning maps for the COCO photo subset.

Check-in 2 pairs each photograph with a synthetic sketch. Canny uses the same
low and high thresholds as the baseline. HED runs only when its OpenCV DNN
prototxt and caffemodel are passed in, because those weights are not stored
in this repository.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_setup import IMAGE_SIZE

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_rgb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def canny_edges(image: Image.Image, low_threshold: int, high_threshold: int) -> Image.Image:
    edges = cv2.Canny(np.array(image), low_threshold, high_threshold)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)


def hed_edges(image: Image.Image, prototxt: Path, caffemodel: Path) -> Image.Image:
    """Run holistically-nested edge detection and return a three-channel map."""
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    array = np.array(image)
    blob = cv2.dnn.blobFromImage(
        array,
        scalefactor=1.0,
        size=(array.shape[1], array.shape[0]),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    hed = net.forward()[0, 0]
    hed = cv2.resize(hed, (array.shape[1], array.shape[0]))
    hed = np.clip(hed * 255.0, 0, 255).astype(np.uint8)
    hed = np.stack([hed, hed, hed], axis=-1)
    return Image.fromarray(hed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make synthetic Canny or HED sketches for COCO photos")
    parser.add_argument("--images", required=True, help="Folder of COCO photographs")
    parser.add_argument("--output", required=True, help="Folder for the edge maps")
    parser.add_argument("--method", choices=("canny", "hed"), default="canny")
    parser.add_argument("--resolution", type=int, default=IMAGE_SIZE)
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--hed-prototxt", default=None, help="HED deploy.prototxt")
    parser.add_argument("--hed-caffemodel", default=None, help="HED pretrained caffemodel")
    args = parser.parse_args()

    if args.method == "hed" and (not args.hed_prototxt or not args.hed_caffemodel):
        raise SystemExit("HED needs --hed-prototxt and --hed-caffemodel. Those weights are not in the repo.")

    source = Path(args.images)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted(source.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = load_rgb(path, args.resolution)
        if args.method == "canny":
            edges = canny_edges(image, args.canny_low, args.canny_high)
        else:
            edges = hed_edges(image, Path(args.hed_prototxt), Path(args.hed_caffemodel))
        edges.save(output / f"{path.stem}.png")
        written += 1
    if written == 0:
        raise SystemExit(f"No photographs found in {source}.")
    print(f"wrote {written} {args.method} edge maps to {output}")


if __name__ == "__main__":
    main()
