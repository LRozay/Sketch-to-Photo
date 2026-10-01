"""Fine-tune a pretrained ControlNet with LoRA on hand-drawn sketches.

Stable Diffusion 1.5 stays frozen. LoRA is added to the ControlNet attention
projections and trained on sketch, photo, and prompt triples resized to 512.
The saved adapter is what baseline_controlnet.py loads with --lora-path.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_setup import CONTROLNET_MODEL_IDS, IMAGE_SIZE, SD15_MODEL_ID

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def find_with_stem(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def to_tensor(path: Path, size: int, normalize: bool) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    array = torch.tensor(list(image.getdata()), dtype=torch.float32).view(size, size, 3) / 255.0
    if normalize:
        array = array * 2.0 - 1.0
    return array.permute(2, 0, 1)


class HandDrawnSketchDataset(Dataset):
    """Pairs a target photo, a hand-drawn sketch, and a one-line prompt."""

    def __init__(self, root: Path, size: int) -> None:
        sketch_dir = root / "sketches"
        image_dir = root / "images"
        prompt_dir = root / "prompts"
        self.size = size
        self.pairs: list[tuple[Path, Path, Path]] = []
        if sketch_dir.is_dir():
            for sketch_path in sorted(sketch_dir.iterdir()):
                if sketch_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                image_path = find_with_stem(image_dir, sketch_path.stem)
                prompt_path = prompt_dir / f"{sketch_path.stem}.txt"
                if image_path is not None and prompt_path.is_file():
                    self.pairs.append((image_path, sketch_path, prompt_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, sketch_path, prompt_path = self.pairs[index]
        return {
            "pixel_values": to_tensor(image_path, self.size, normalize=True),
            "conditioning": to_tensor(sketch_path, self.size, normalize=False),
            "prompt": prompt_path.read_text().strip(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune ControlNet on hand-drawn sketches")
    parser.add_argument("--data-dir", required=True, help="Folder with images/, sketches/, and prompts/")
    parser.add_argument("--control-type", choices=sorted(CONTROLNET_MODEL_IDS), default="scribble")
    parser.add_argument("--output-dir", default="outputs/controlnet-lora")
    parser.add_argument("--resolution", type=int, default=IMAGE_SIZE)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=261)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen_dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(SD15_MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD15_MODEL_ID, subfolder="text_encoder").to(device, dtype=frozen_dtype)
    vae = AutoencoderKL.from_pretrained(SD15_MODEL_ID, subfolder="vae").to(device, dtype=frozen_dtype)
    unet = UNet2DConditionModel.from_pretrained(SD15_MODEL_ID, subfolder="unet").to(device, dtype=frozen_dtype)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_IDS[args.control_type]).to(device)

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.eval()
    vae.eval()
    unet.eval()

    controlnet.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    trainable = [param for param in controlnet.parameters() if param.requires_grad]
    if not trainable:
        raise SystemExit("LoRA did not attach any trainable ControlNet parameters.")

    noise_scheduler = DDPMScheduler.from_pretrained(SD15_MODEL_ID, subfolder="scheduler")
    dataset = HandDrawnSketchDataset(Path(args.data_dir), args.resolution)
    if len(dataset) == 0:
        raise SystemExit(
            f"No sketch, photo, and prompt triples found in {args.data_dir}. "
            "Expected images/, sketches/, and prompts/ with the same filename stem."
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    controlnet.train()
    step = 0
    data_iter = iter(loader)
    while step < args.train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(device, dtype=frozen_dtype)
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            tokens = tokenizer(
                list(batch["prompt"]),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            encoder_hidden_states = text_encoder(tokens)[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        conditioning = batch["conditioning"].to(device, dtype=torch.float32)

        control_out = controlnet(
            noisy_latents.float(),
            timesteps,
            encoder_hidden_states=encoder_hidden_states.float(),
            controlnet_cond=conditioning,
            conditioning_scale=args.conditioning_scale,
        )
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[sample.to(dtype=frozen_dtype) for sample in control_out.down_block_res_samples],
            mid_block_additional_residual=control_out.mid_block_res_sample.to(dtype=frozen_dtype),
        ).sample

        loss = F.mse_loss(model_pred.float(), noise.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        if step == 1 or step % 50 == 0 or step == args.train_steps:
            print(f"step {step}/{args.train_steps} loss {loss.item():.4f}")

    output_dir = Path(args.output_dir) / args.control_type
    output_dir.mkdir(parents=True, exist_ok=True)
    controlnet.save_lora_adapter(output_dir)
    print(f"saved ControlNet LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
