"""Shared Stable Diffusion 1.5 and ControlNet settings for check-in 1."""

SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Pretrained ControlNet checkpoints used as the baseline before LoRA fine-tuning.
CONTROLNET_MODEL_IDS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "scribble": "lllyasviel/sd-controlnet-scribble",
}

# Photos, synthetic edge maps, and hand-drawn sketches are resized to this square.
IMAGE_SIZE = 512
