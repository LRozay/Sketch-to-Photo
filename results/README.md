# Experiment results

Baseline FID and CLIP score are recorded here after the HPC jobs finish.
`baseline_metrics.json` starts with null values on purpose. Fill it with
`src/record_baseline_metrics.py` only when real generated folders exist.

`experiment_matrix.json` lists the three scored configurations from Check-in 3
Table 4: pretrained canny, pretrained scribble, and scribble plus LoRA. Scores
stay null until `src/record_config_metrics.py` and `src/summarize_survey.py`
run on real outputs.

