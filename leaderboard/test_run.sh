# Parameterization settings. These will be explained in 2.2. Now simply copy them to run the test.
export ROUTES=${LEADERBOARD_ROOT}/../data/1_scenario_per_route_v5/training_1_scenario/ControlLoss/2438.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=${LEADERBOARD_ROOT}/leaderboard/autoagents/human_agent.py # for the keyboard
# export TEAM_AGENT=${LEADERBOARD_ROOT}/leaderboard/autoagents/human_agent_steering_wheel.py
export CHECKPOINT_ENDPOINT=${LEADERBOARD_ROOT}/../data/1_scenario_per_route_v5/training_1_scenario/ControlLoss/2438.json
export CHALLENGE_TRACK_CODENAME=SENSORS

./scripts/run_evaluation.sh
