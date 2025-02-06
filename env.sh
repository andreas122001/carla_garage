export CARLA_ROOT=/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20
export WORK_DIR=/cluster/work/andrebw/carla_garage
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}


# export REPETITIONS=1
# export DEBUG_CHALLENGE=0

export PTH_ROUTE=${WORK_DIR}/leaderboard/data/routes_devtest

# Start the carla server
export HOST="localhost"
export PORT=2000

# export TEAM_AGENT=${WORK_DIR}/team_code/data_agent.py # use autopilot.py here to only run the expert without data generation
# export CHALLENGE_TRACK_CODENAME=MAP
# export ROUTES=${PTH_ROUTE}.xml    
# export TM_PORT=$((PORT + 3))

# export CHECKPOINT_ENDPOINT=${PTH_ROUTE}.json
# export TEAM_CONFIG=${PTH_ROUTE}.xml
# export PTH_LOG='datasets/dataset_test1/data/data'
# export RESUME=1
# export DATAGEN=1
# export SAVE_PATH='datasets/dataset_test1/data/data'
# export TM_SEED=0
export FREE_WORLD_PORT=$PORT

echo "Env set."