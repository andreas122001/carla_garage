import gzip
import os
from time import sleep
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from team_code.config import GlobalConfig


root = "/cluster/work/andrebw/carla_garage/database/dataset_4hz_2024_10_26/data"

buckets = GlobalConfig().target_speed_bins
target_speeds = []
target_speed_classes = []

sub_roots = [os.path.join(root_, "measurements") for root_, dirs, _ in os.walk(root) if "measurements" in dirs]
for sub_root in tqdm(sub_roots, position=0):
    for file_ in tqdm(os.listdir(sub_root), position=1, leave=False):
        with gzip.open(os.path.join(sub_root, file_), mode='rb') as f:
            ts = json.loads(f.read())['speed']
            ts_class = np.digitize(x=ts, bins=buckets)

            target_speeds.append(ts)
            target_speed_classes.append(ts_class)

print(pd.DataFrame(target_speeds).value_counts())
print(pd.DataFrame(target_speed_classes).value_counts())
print(compute_class_weight(class_weight='balanced', classes=np.unique(target_speed_classes), y=target_speed_classes))
print(np.unique(target_speed_classes))

print("Max speed:", np.max(target_speeds))
