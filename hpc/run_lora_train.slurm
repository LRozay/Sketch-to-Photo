#!/bin/bash
#SBATCH --job-name=s2p-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/lora_%j.out
#SBATCH --error=outputs/logs/lora_%j.err

# First rank-8 LoRA run on scribble ControlNet. Stable Diffusion 1.5 stays frozen.
# Default is 500 steps, batch size 1, seed 261, matching train_controlnet_lora.py.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p outputs/logs outputs/controlnet-lora

module load python/3.11 cuda/12.1 || true
source .venv/bin/activate

python src/check_hand_drawn_data.py --data-dir data/hand_drawn
python src/train_controlnet_lora.py \
  --data-dir data/hand_drawn \
  --control-type scribble \
  --output-dir outputs/controlnet-lora \
  --train-steps 500 \
  --lora-rank 8 \
  --batch-size 1 \
  --conditioning-scale 1.0 \
  --seed 261

echo "first LoRA adapter saved under outputs/controlnet-lora/scribble"
