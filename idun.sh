#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --job-name=tfpp-pdm_lite-max_speed21
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/%x/out-%j.out
#SBATCH --error=logs/%x/err-%j.out
#SBATCH --time=4-00:00:00
#SBATCH --constraint="a100"
# #SBATCH --nodelist=idun-07-05

#export MASTER_ADDR=localhost

# Run training script
cd /cluster/work/andrebw/carla_carage_main

export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=1

sh run_training.sh
