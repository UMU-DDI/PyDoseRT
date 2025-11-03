#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-504 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-01:00:00
#SBATCH --error=/cephyr/users/attilas/Alvis/out/%J_error.out
#SBATCH --output=/cephyr/users/attilas/Alvis/out/%J_output.out

module --ignore-cache load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /cephyr/users/attilas/Alvis/data/newenv/newenv/bin/activate

python3 scripts/static_optimize_single_patient.py
wait
