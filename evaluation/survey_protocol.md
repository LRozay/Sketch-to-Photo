# Pairwise sketch-following survey

The survey asks which generated photo follows the sketch more closely. CLIP
score only measures prompt agreement, so this preference check is the sketch
fidelity view in the evaluation plan.

## What each rater sees

For every complete example the sheet lists:

1. the hand-drawn sketch
2. the short text prompt
3. the pretrained scribble baseline photo
4. the LoRA-adapted photo

Both photos are produced with the same seed, conditioning scale, and guidance
scale so the choice is about the model, not a new noise draw.

## Rating rule

In the `follows_sketch_better` column, mark one of:

- `baseline`
- `lora`
- `tie`

Judge sketch following first. Do not prefer an image only because it looks more
photographic if it invents objects that are absent from the sketch.

## How to build the blank sheet

```bash
python src/evaluate_generations.py survey \
  --sketches data/hand_drawn/sketches \
  --prompts data/hand_drawn/prompts \
  --baseline outputs/compare/baseline \
  --lora outputs/compare/lora \
  --output outputs/pairwise_survey.csv
```

Filled responses belong under `results/` after collection. Do not invent votes.
