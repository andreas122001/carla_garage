import os
from data import CARLA_Data
# from bench2drive_data import Bench2Drive_Data as CARLA_Data
from config import GlobalConfig
import numpy as np
import torch
import torch.nn.functional as F

print("python test.py")

# data_root = "/cluster/work/andrebw/database/bench2drive"
data_root = "/cluster/work/andrebw/carla_garage/database/dataset_4hz_2024_10_26/data"
config = GlobalConfig()
config.initialize(
    root_dir=data_root,
    setting="all",
    num_repetitions=1,
    augment=0,
    no_auto_data=0# 1
)


# train_data = [
#     os.path.join(config.root_dir, dir_) for dir_ in os.listdir(config.root_dir)
# ][:1]

train_data = config.train_data[:1]

print(train_data[0])
train_set = CARLA_Data(
    root=train_data,
    config=config,
    estimate_class_distributions=False,
    estimate_sem_distribution=False,
    shared_dict=None,
    rank=0,
)

print(list(train_set[0].keys()))
for i in range(len(train_set)):
    lidar = train_set[i]["lidar"]
    print(lidar.shape)
    scaled = F.interpolate(torch.tensor(lidar).unsqueeze(0), size=(512,512), mode='bilinear', align_corners=False)
    scaled = torch.round(scaled, decimals=1)
    print(scaled[np.where(scaled != 0)])
        # print(lidar[np.where(lidar != 0)])
        # print(scaled.)


