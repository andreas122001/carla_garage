from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import os
from time import sleep
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import ujson
from team_code.config import GlobalConfig


root = "/cluster/work/andrebw/carla_garage/database/dataset_4hz_2024_10_26/data"
buckets = GlobalConfig().target_speed_bins

def process_file(filepath):
    """Reads and processes a single file, extracting target speed and its class."""

    with gzip.open(filepath, mode='rb') as f:
        measurements = json.loads(f.read())
        ts = measurements['target_speed']
        # If we are braking, set class to 0
        if measurements['brake']:
            ts_class = 0
        # Else, we use the class, but add one for 'not brake'
        else:
            ts_class = np.digitize(x=ts, bins=buckets) + 1
        return ts, ts_class

def get_measurement_files(root):
    """Generates a list of all measurement files in the given root directory."""
    sub_roots = [os.path.join(root_) for root_, dirs, _ in os.walk(root) if "measurements" in dirs]
    measurement_files = []
    for sub_root in tqdm(sub_roots):
        # If no results, drop it
        if not os.path.isfile(sub_root + '/results.json.gz'):
            continue
        
        # if results are bad, drop it
        with gzip.open(sub_root + '/results.json.gz', 'rt', encoding='utf-8') as f:
            results_route = ujson.load(f)
        if results_route['scores']['score_composed'] < 100.0:
            continue
        
        # Add all measurement files from this route
        for file_ in os.listdir(os.path.join(sub_root, "measurements")):
            measurement_files.append(os.path.join(sub_root, "measurements", file_))
    return measurement_files

# Gather all file paths to process
measurement_files = get_measurement_files(root)

# Use multiprocessing to speed up file processing
target_speeds = []
target_speed_classes = []

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_file, file) for file in measurement_files]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
        ts, ts_class = future.result()
        if ts is not None:
            ts = np.digitize(x=ts, bins=range(24))
            target_speeds.append(ts)
            target_speed_classes.append(ts_class)

# Compute statistics and display results
weights = compute_class_weight(class_weight='balanced', classes=np.unique(target_speed_classes), y=target_speed_classes)
# print(pd.DataFrame(target_speeds).value_counts())
print(pd.DataFrame(target_speed_classes).value_counts())
print("    self.target_speed_weights = [")
for w in weights:
    print(f"      {w},")
print("    ]")
print(np.unique(target_speed_classes))
print("Max speed:", np.max(target_speeds))
plt.plot(np.array(target_speeds))
plt.savefig("target_speed_distributions.png")
