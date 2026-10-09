# Sketch-to-Photo

CMPE 261 check-in 1 setup for sketch-to-photo synthesis with ControlNet-guided diffusion.

The baseline is Stable Diffusion 1.5 with the pretrained ControlNet canny and scribble checkpoints loaded through Hugging Face Diffusers. Photos and conditioning images are resized to 512 x 512. ControlNet is fine-tuned with LoRA on a small set of real hand-drawn sketches. The ControlNet conditioning scale and the text guidance scale stay configurable for the ablations in the project plan.

`src/model_setup.py` records the Stable Diffusion 1.5 model id, the canny and scribble ControlNet checkpoint ids, and the 512 image size. `src/baseline_controlnet.py` runs that pretrained baseline.

## Baseline inference

```bash
python src/baseline_controlnet.py \
  --sketch path/to/sketch.png \
  --prompt "a red bicycle leaning against a wall" \
  --control-type scribble \
  --conditioning-scale 1.0 \
  --guidance-scale 7.5
```

For a canny edge map, add `--apply-canny`. Lower Canny thresholds keep thin and broken contours and also keep more noise. Higher thresholds make a cleaner edge map and drop some weak detail. Pass an existing edge map without `--apply-canny` when the edges were already built. The scribble checkpoint is the one used for freehand sketches.

Each run also writes a `.txt` file next to the image with the prompt, sketch path, control type, conditioning scale, guidance scale, step count, and seed. The conditioning image ControlNet actually saw is saved beside it as `<name>_condition.png`.

Pass `--lora-path outputs/controlnet-lora/scribble` after a fine-tuning run to generate with that adapter on the same sketch.

## Baseline ablation

`src/run_baseline_ablation.py` repeats the pretrained baseline over conditioning scales 0.5, 1.0, and 1.5 and guidance scales 5.0, 7.5, and 12.0. That is the check-in 1 comparison before LoRA fine-tuning.

```bash
python src/run_baseline_ablation.py \
  --sketch path/to/sketch.png \
  --prompt "a red bicycle leaning against a wall" \
  --control-type scribble \
  --output-dir outputs/ablation
```

## LoRA fine-tuning

`src/train_controlnet_lora.py` freezes Stable Diffusion 1.5 and trains LoRA on the ControlNet attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`). Training expects the folders under `data/hand_drawn/`, with matching filenames:

```text
data/hand_drawn/images/   photo targets
data/hand_drawn/sketches/ hand-drawn sketches
data/hand_drawn/prompts/  one text prompt per sketch
```

The hand-drawn set is the 50 to 100 sketches from the proposal. Each sketch keeps the short prompt used as the text condition. `src/check_hand_drawn_data.py` lists sketches that are missing a photo or a prompt before training starts.

```bash
python src/check_hand_drawn_data.py --data-dir data/hand_drawn
```

```bash
python src/train_controlnet_lora.py \
  --data-dir data/hand_drawn \
  --control-type scribble \
  --conditioning-scale 1.0 \
  --output-dir outputs/controlnet-lora
```

The adapter is written under the output directory for that control type. Training and batch inference are intended for the COE HPC machines.
