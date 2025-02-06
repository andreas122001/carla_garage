#!/bin/bash
#SBATCH --job-name=0-pdm_lite_full
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH -o evaluation/pdm_lite_full_model_0030/pdm_lite_full/run_files/logs/qsub_out0.log
#SBATCH -e evaluation/pdm_lite_full_model_0030/pdm_lite_full/run_files/logs/qsub_err0.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20gb
#SBATCH --time=01-12:00:00
#SBATCH --gres=gpu:1
# #SBATCH --constraint=(a100)

FREE_WORLD_PORT=`comm -23 <(seq 10000 10400 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`


echo 'World Port:' $FREE_WORLD_PORT


FREE_STREAMING_PORT=`comm -23 <(seq 20000 20400 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`


echo "Launching CARLA with world-port=$FREE_WORLD_PORT and streaming-port=$FREE_STREAMING_PORT"


sh /cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &


sleep 20


chmod u+x evaluation/pdm_lite_full_model_0030/pdm_lite_full_e0_model_0030/run_bashs/eval_routes_validation.sh


./evaluation/pdm_lite_full_model_0030/pdm_lite_full_e0_model_0030/run_bashs/eval_routes_validation.sh $FREE_WORLD_PORT


sleep 2

