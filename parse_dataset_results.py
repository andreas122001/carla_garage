import glob
import gzip
import numpy as np
from tqdm import tqdm
import ujson

# root = "/cluster/work/andrebw/carla_garage/database/dataset_default_expert2024_11_08/data"
# root = "/cluster/work/andrebw/carla_garage/database/eval_default_scenarios/data"
root = "/cluster/work/andrebw/carla_garage/database/eval_pdml_scenarios/data"

result_files = glob.glob(f"{root}/**/results.json.gz", recursive=True)

# Scenarios
results = {}
for result_path in tqdm(result_files):
    scenario = result_path.split("/")[-4]
    if scenario not in results.keys():
        results[scenario] = []
    with gzip.open(result_path, "rt", encoding="utf-8") as f:
        results_route = ujson.load(f)
        results[scenario].append(results_route["scores"])

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
save_path = "/cluster/work/andrebw/carla_garage_main/eval_pdml_town13.json"
print(Path(save_path))
with open(save_path, "w") as f:
    ujson.dump(results, f, indent=4)

