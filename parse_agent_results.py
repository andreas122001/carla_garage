import glob
import gzip
import numpy as np
from tqdm import tqdm
import ujson
import os

scenarios_root = "/cluster/work/andrebw/carla_garage/data/50x36_town_13"
_,scenarios,_ = next(os.walk(scenarios_root))

route_to_scenario = {}
for scenario in scenarios: 
    for route in os.listdir(os.path.join(scenarios_root, scenario)):
        if route.endswith(".xml"):
            route_to_scenario[route[:-4]] = scenario

agent = "tfpp-pdm_lite-max_speed21_lb2"
root = f"/cluster/work/andrebw/carla_garage_main/evaluation/{agent}_model_0030/{agent}_e0_model_0030/results"
result_files = glob.glob(f"{root}/**/*.json", recursive=True)

# Scenarios
results = {}
for result_path in tqdm(result_files):
    scenario = route_to_scenario[result_path.split("/")[-1][:-5]]
    if scenario not in results.keys():
        results[scenario] = []
    with open(result_path, "rt", encoding="utf-8") as f:
        results_route = ujson.load(f)
        if results_route["_checkpoint"]["global_record"].get("scores_mean", -1) != -1:
            results[scenario].append(results_route["_checkpoint"]["global_record"]["scores_mean"])

# Post-process 
for scenario, scenario_res in results.items():
    score_route = [res["score_route"] for res in scenario_res]
    score_penalty = [res["score_penalty"] for res in scenario_res]
    score_composed = [res["score_composed"] for res in scenario_res]
    # Just overwrite with the averages
    results[scenario] = {
        "scores_mean": {
            "score_route": np.mean(score_route),
            "score_penalty": np.mean(score_penalty),
            "score_composed": np.mean(score_composed)
        },
        "scores_std": {
            "score_route": np.std(score_route),
            "score_penalty": np.std(score_penalty),
            "score_composed": np.std(score_composed)
        }
    }

scores_route = [entry["scores_mean"]["score_route"] for entry in results.values()]
score_penalty = [entry["scores_mean"]["score_penalty"] for entry in results.values()]
score_composed = [entry["scores_mean"]["score_composed"] for entry in results.values()]

results['Total'] = {
    "scores_mean": {
        "score_route": np.mean(scores_route),
        "score_penalty": np.mean(score_penalty),
        "score_composed": np.mean(score_composed)
    },
    # Notice, the std here is across scenarios, not routes in total
    "scores_std": {
        "score_route": np.std(scores_route),
        "score_penalty": np.std(score_penalty),
        "score_composed": np.std(score_composed)
    },
}


from pathlib import Path
save_path = f"/cluster/work/andrebw/carla_garage_main/{agent}_results.json"
print(Path(save_path))
with open(save_path, "w") as f:
    ujson.dump(results, f, indent=4)


