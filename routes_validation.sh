#!/bin/bash
export SCENARIO_RUNNER_ROOT=/cluster/work/andrebw/carla_garage/scenario_runner
export LEADERBOARD_ROOT=/cluster/work/andrebw/carla_garage/leaderboard

# carla
export CARLA_ROOT=/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20
export CARLA_SERVER=/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20/PythonAPI
export PYTHONPATH=$PYTHONPATH:/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:
export PYTHONPATH=$PYTHONPATH:$SCENARIO_RUNNER_ROOT

export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT=/cluster/work/andrebw/carla_garage/team_code/autopilot.py
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES=/cluster/work/andrebw/carla_garage/leaderboard/data/routes_validation.xml
export TM_SEED=0

export CHECKPOINT_ENDPOINT=/cluster/work/andrebw/carla_garage/datasets/dataset_test1/results/data/routes_validation_result.json
export TEAM_CONFIG=/cluster/work/andrebw/carla_garage/leaderboard/data/routes_validation.xml
export RESUME=1
export DATAGEN=1
export SAVE_PATH=/cluster/work/andrebw/carla_garage/datasets/dataset_test1/data/data

echo "Loading modules..."
# module load CUDA/12.1.1
module load Anaconda3/2024.02-1
module load libjpeg-turbo/3.0.1-GCCcore-13.2.0

echo "Activating conda environment"
conda activate pdm_lite
conda env list

export FREE_STREAMING_PORT=$1
export FREE_WORLD_PORT=$2
export TM_PORT=$3

echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
echo "TM_PORT: $TM_PORT"

echo "Launching CARLA with world-port=$FREE_WORLD_PORT and streaming-port=$FREE_STREAMING_PORT"
sh /cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &

sleep 20

nvidia-smi
echo "Which python: $(which python)"
echo "Carla API version: $(pip freeze | grep carla)"

echo "Running python script..."
python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${FREE_WORLD_PORT} --traffic-manager-port=${TM_PORT} --traffic-manager-seed=${TM_SEED} --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=120    

echo "Finished!"
