# Results and discussion outline

These sections stay structural until the HPC runs finish. Numbers and survey
counts are left blank on purpose so the report does not invent scores.

## 7. Results

### 7.1 Baseline on hand-drawn sketches

- Model: pretrained scribble ControlNet, seed 261, 30 UniPC steps
- Default scales: conditioning 1.0, guidance 7.5
- Batch outputs: `outputs/baseline/`
- CLIP score (hand-drawn prompts): _pending_
- FID vs held-out COCO photographs: _pending_
- Metrics file: `results/baseline_metrics.json`

### 7.2 First LoRA adapter

- Rank 8, alpha 8, 500 steps, AdamW 1e-4
- Adapter path: `outputs/controlnet-lora/scribble/`
- Training status: _not started on HPC yet_

### 7.3 Fixed-seed side-by-side comparison

- Same sketches and prompts for baseline and LoRA
- Folders: `outputs/compare/baseline/` and `outputs/compare/lora/`
- Insert nine-cell grids for baseline and for the adapter after the scale sweeps

### 7.4 Human preference

- Sheet: `outputs/pairwise_survey.csv`
- Question: which photo follows the sketch more closely
- Preference counts: _pending_

## 8. Discussion and failure cases

Use the real grids to discuss:

1. ignored thin strokes on freehand input
2. extra objects invented to satisfy the prompt
3. copied strokes that hurt realism after fine-tuning on a small set

Draft claim to test later: LoRA should improve sketch following on messy
strokes without fully solving prompt-sketch conflicts. Do not write the final
claim until the comparison images and survey votes exist.

## Still open for later check-ins

- Finish the 50 to 100 hand-drawn triples and the full COCO subset on disk
- Record the first real FID and CLIP numbers
- Complete the LoRA training run and fill the preference table
- Write the conclusion after those measurements exist
