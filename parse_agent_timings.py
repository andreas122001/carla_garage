import glob
import gzip
import numpy as np
from tqdm import tqdm
import ujson

# root = "/cluster/work/andrebw/carla_garage/database/dataset_default_expert2024_11_08/data"
root = "/cluster/work/andrebw/carla_garage/database/dataset_4hz_2024_10_26/data"
# root = "/cluster/work/andrebw/carla_garage/database/eval_default_scenarios/data"
# root = "/cluster/work/andrebw/carla_garage/database/eval_pdml_scenarios/data"

result_files = glob.glob(f"{root}/**/results.json.gz", recursive=True)

results = []
total_routes = len(result_files)
perfect_routes = 0
for result_path in tqdm(result_files):
    with gzip.open(result_path, "rt", encoding="utf-8") as f:
        results_route = ujson.load(f)
        if results_route["scores"]["score_composed"] == 100:
            perfect_routes += 1
            results.append(results_route["meta"]["duration_game"])
results = np.array(results)
print(results.mean(), results.std())
print(f"Perfect routes: {perfect_routes}/{total_routes}")


# from pathlib import Path
# save_path = "/cluster/work/andrebw/carla_garage_main/eval_pdml_town13.json"
# print(Path(save_path))
# with open(save_path, "w") as f:
#     ujson.dump(results, f, indent=4)
