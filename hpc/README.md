# College of Engineering HPC jobs

Training and batch inference need a GPU node. Submit from the repository root
after the hand-drawn triples and the Python environment are ready.

```bash
sbatch hpc/run_baseline.slurm
sbatch hpc/run_lora_train.slurm
```

Both jobs keep seed 261. The baseline job writes `outputs/baseline/`. The LoRA
job writes the rank-8 adapter under `outputs/controlnet-lora/scribble/`.
