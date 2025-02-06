#!/bin/bash

# This script starts PDM-Lite and the CARLA simulator on a local machine

# Exports
. ./env.sh

# carla
export REPETITIONS=1
export DEBUG_CHALLENGE=0

export PTH_ROUTE=${WORK_DIR}/leaderboard/data/routes_devtest

# Start the carla server
export HOST="localhost"
export PORT=2000
echo "Connection: $HOST:$PORT"

export TEAM_AGENT=${WORK_DIR}/team_code/data_agent.py # use autopilot.py here to only run the expert without data generation
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES=${PTH_ROUTE}.xml
export TM_PORT=$((PORT + 3))

export CHECKPOINT_ENDPOINT=${PTH_ROUTE}.json
export TEAM_CONFIG=${PTH_ROUTE}.xml
export PTH_LOG='logs'
export RESUME=1
export DATAGEN=1
export SAVE_PATH='logs'
export TM_SEED=0

echo "Running leaderboard evaluator..."
# Start the actual evaluation / data generation
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
       	--host=${HOST}\
       	--port=${PORT}\
       	--traffic-manager-port=${TM_PORT}\
       	--routes=${ROUTES}\
       	--repetitions=${REPETITIONS}\
       	--track=${CHALLENGE_TRACK_CODENAME}\
       	--checkpoint=${CHECKPOINT_ENDPOINT}\
       	--agent=${TEAM_AGENT}\
       	--agent-config=${TEAM_CONFIG}\
       	--debug=0\
       	--resume=${RESUME}\
       	--timeout=2000\
       	--traffic-manager-seed=${TM_SEED}
