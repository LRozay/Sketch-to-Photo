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
