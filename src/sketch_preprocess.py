"""Prepare hand-drawn scribbles for the pretrained ControlNet scribble checkpoint.

Check-in 3 Section 8.4 notes that the scribble ControlNet was trained on white
strokes on a black background, while paper drawings are dark strokes on white.
An uninverted sketch would look almost empty to that checkpoint. Threshold the
dark strokes and invert them before ControlNet sees the image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def invert_scribble_for_controlnet(image: Image.Image, threshold: int = 200) -> Image.Image:
    """Turn dark-on-light paper strokes into white-on-black ControlNet scribbles."""
    gray = np.array(image.convert("L"))
    strokes = (gray < threshold).astype(np.uint8) * 255
    rgb = np.stack([strokes, strokes, strokes], axis=-1)
    return Image.fromarray(rgb)
