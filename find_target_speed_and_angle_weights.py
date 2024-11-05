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

config = GlobalConfig()
root = "/cluster/work/andrebw/carla_garage/database/dataset_4hz_2024_10_26/data"
speed_bins = config.target_speed_bins
angle_bins = config.angle_bins

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


def process_file(filepath):
    """Reads and processes a single file, extracting target speed and angle classes."""

    with gzip.open(filepath, mode='rb') as f:
        measurements = json.loads(f.read())
        ts = measurements['target_speed']
        angle = measurements['angle']
        # If we are braking, set class to 0
        if measurements['brake']:
            ts_class = 0
        # Else, we use the class, but add one for 'not brake'
        else:
            ts_class = np.digitize(x=ts, bins=speed_bins) + 1
        angle_class = np.digitize(x=angle, bins=config.angle_bins)

        return ts_class, angle_class

# Gather all file paths to process
measurement_files = get_measurement_files(root)

# Use multiprocessing to speed up file processing
target_speed_classes = []
angle_classes = []

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_file, file) for file in measurement_files]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
        ts_class, angle_class = future.result()
        if not None in [ts_class, angle_class]:
            target_speed_classes.append(ts_class)
            angle_classes.append(angle_class)

# Compute statistics and display results
speed_weights = compute_class_weight(class_weight='balanced', classes=np.unique(target_speed_classes), y=target_speed_classes)
angle_weights = compute_class_weight(class_weight='balanced', classes=np.unique(angle_classes), y=angle_classes)
print(pd.DataFrame(target_speed_classes).value_counts())
print(pd.DataFrame(angle_classes).value_counts())
print("    self.target_speed_weights = [")
for w in speed_weights:
    print(f"      {w},")
print("    ]")
print("    self.angle_weights = [")
for w in angle_weights:
    print(f"      {w},")
print("    ]")
