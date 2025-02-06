export CARLA_ROOT=/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=/cluster/work/andrebw/carla_garage/scenario_runner
export LEADERBOARD_ROOT=/cluster/work/andrebw/carla_garage/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export PORT=2300
echo 'World Port:' $PORT
export TM_PORT=`comm -23 <(seq 32300 32700 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'TM Port:' $TM_PORT
# export ROUTES=/cluster/work/andrebw/carla_garage/data/50x36_town_13/HighwayExit/1396_0.xml
export ROUTES=/cluster/work/andrebw/carla_garage/data/50x36_town_13/ConstructionObstacle/1271_1.xml
# export ROUTES=/cluster/work/andrebw/carla_garage/data/50x38_town_12/HighwayExit/4081_0.xml
export TEAM_AGENT=team_code/sensor_agent.py
export TEAM_CONFIG=logs/tfpp-pdm_lite-max_speed21/
export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export RESUME=0
export CHECKPOINT_ENDPOINT="evaluation/test2/result.json"
export DEBUG_CHALLENGE=1
export DATAGEN=0
export SAVE_PATH="evaluation/test2/logs"
export DIRECT=0
export UNCERTAINTY_WEIGHT=1
export UNCERTAINTY_THRESHOLD=0.5
export HISTOGRAM=0
export BLOCKED_THRESHOLD=180
export TMP_VISU=0
export VISU_PLANT=0
export SLOWER=1
export STOP_CONTROL=1
export TP_STATS=0
export BENCHMARK=lav

echo "Loading modules..."
module purge
module load Anaconda3/2024.02-1
module load libjpeg-turbo/2.1.5.1-GCCcore-12.3.0
module load FFmpeg/6.0-GCCcore-12.3.0

echo "Activating conda environment..."
conda activate garage
conda env list

rm "evaluation/test2/result.json"

python3 -u ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=1 --record=${RECORD_PATH} --resume=${RESUME} --port=${PORT} --timeout=6000 --traffic-manager-port=${TM_PORT}

