#!/bin/bash
#SBATCH --job-name=s2p-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/baseline_%j.out
#SBATCH --error=outputs/logs/baseline_%j.err

# End-to-end pretrained scribble baseline on the hand-drawn triples.
# Seed stays 261 so later LoRA comparisons reuse the same noise draw.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p outputs/logs outputs/baseline

module load python/3.11 cuda/12.1 || true
source .venv/bin/activate

python src/check_hand_drawn_data.py --data-dir data/hand_drawn
python src/run_batch_baseline.py \
  --data-dir data/hand_drawn \
  --control-type scribble \
  --output-dir outputs/baseline \
  --conditioning-scale 1.0 \
  --guidance-scale 7.5 \
  --steps 30 \
  --seed 261

echo "baseline batch finished"
