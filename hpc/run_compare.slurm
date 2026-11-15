#!/bin/bash
#SBATCH --job-name=s2p-compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/compare_%j.out
#SBATCH --error=outputs/logs/compare_%j.err

# Fixed-seed baseline versus LoRA comparison on the hand-drawn set.
# Requires the rank-8 adapter from hpc/run_lora_train.slurm.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p outputs/logs outputs/compare/baseline outputs/compare/lora

module load python/3.11 cuda/12.1 || true
source .venv/bin/activate

ADAPTER="outputs/controlnet-lora/scribble"
if [[ ! -d "$ADAPTER" ]]; then
  echo "Missing LoRA adapter at $ADAPTER. Submit hpc/run_lora_train.slurm first."
  exit 1
fi

python src/check_hand_drawn_data.py --data-dir data/hand_drawn
python src/compare_baseline_lora.py \
  --data-dir data/hand_drawn \
  --control-type scribble \
  --lora-path "$ADAPTER" \
  --baseline-dir outputs/compare/baseline \
  --lora-dir outputs/compare/lora \
  --conditioning-scale 1.0 \
  --guidance-scale 7.5 \
  --steps 30 \
  --seed 261

python src/evaluate_generations.py survey \
  --sketches data/hand_drawn/sketches \
  --prompts data/hand_drawn/prompts \
  --baseline outputs/compare/baseline \
  --lora outputs/compare/lora \
  --output outputs/pairwise_survey.csv

echo "fixed-seed comparison and blank survey sheet finished"
