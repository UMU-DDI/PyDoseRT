#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-504 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02-00:00:00
#SBATCH --error=/cephyr/users/attilas/Alvis/out/%J_error.out
#SBATCH --output=/cephyr/users/attilas/Alvis/out/%J_output.out

module load virtualenv/20.23.1-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
# module --ignore-cache load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /cephyr/users/attilas/Alvis/data/attila/autoenv/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python3 -u scripts/rtplan_test_1arc.py
wait
