#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --job-name=test
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00
# #SBATCH --nodelist=idun-07-01
#SBATCH --constraint=(a100|v100)

FREE_WORLD_PORT=$(comm -23 <(seq {carla_world_port_start} {carla_world_port_start + 400} | sort) < (ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1)

sh /cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 &

sleep 20

sh test.sh
