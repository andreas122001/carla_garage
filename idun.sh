#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --job-name=tfpp-default-max_speed21
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/%x/log.out
#SBATCH --error=logs/%x/log.out
#SBATCH --time=4-00:00:00
#SBATCH --constraint="a100"
# #SBATCH --nodelist=idun-07-05

#export MASTER_ADDR=localhost

# Run training script
cd /cluster/work/andrebw/carla_garage_main

export OMP_NUM_THREADS=12

sh run_training.sh
