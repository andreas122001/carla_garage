from typing import Dict
from data import CARLA_Data

import os
import ujson
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from tqdm import tqdm
import sys
import cv2
import gzip
import laspy
import io
import transfuser_utils as t_u
import gaussian_target as g_t
import random
from sklearn.utils.class_weight import compute_class_weight
from center_net import angle2class
from imgaug import augmenters as ia

# Code copied from team_code/data.py and adapted for Bench2Drive dataset

class Bench2Drive_Data(CARLA_Data):
    def __init__(
        self,
        root,
        config,
        estimate_class_distributions=False,
        estimate_sem_distribution=False,
        shared_dict=None,
        rank=0,
    ):
        super().__init__(
            root,
            config,
            estimate_class_distributions,
            estimate_sem_distribution,
            shared_dict,
            rank,
        )

    def _load_index(
        self,
        root,
        config,
        estimate_class_distributions=False,
        estimate_sem_distribution=False,
        rank=0,
    ):
        rgb_dir = "camera/rgb_front"
        lidar_dir = "lidar"
        semantic_dir = "camera/semantic_front"
        depth_dir = "camera/depth_front"

        # Iterate over each route and gather the results
        for route_dir in tqdm(
            root, file=sys.stdout, disable=rank != 0 or len(root) < 1
        ):
            num_seq = len(
                os.listdir(os.path.join(route_dir, lidar_dir))
            )  # check length of lidar dir

            # Iterate over all datapoints in the route
            for seq in range(
                config.skip_first,
                num_seq - self.config.pred_len - self.config.seq_len,
            ):
                if seq % config.train_sampling_rate != 0:
                    continue

                # load input seq and pred seq jointly
                image = []
                semantic = []
                depth = []
                lidar = []
                box = []
                future_box = []
                measurement = []

                # Loads the current (and past) frames (if seq_len > 1)
                for idx in range(self.config.seq_len):
                    image.append(
                        os.path.join(route_dir, rgb_dir, f"{(seq + idx):05}.jpg")
                    )
                    semantic.append(
                        os.path.join(route_dir, semantic_dir, f"{(seq + idx):05}.png")
                    )
                    depth.append(
                        os.path.join(route_dir, depth_dir, f"{(seq + idx):05}.png")
                    )
                    lidar.append(
                        os.path.join(route_dir, lidar_dir, f"{(seq + idx):05}.laz")
                    )
                    box.append(
                        os.path.join(route_dir, "anno", f"{(seq + idx):05}.json.gz")
                    )
                    forcast_step = int(
                        config.forcast_time / (config.data_save_freq / config.carla_fps)
                        + 0.5
                    )
                    future_box.append(
                        os.path.join(
                            route_dir,
                            "anno",
                            f"{(seq + idx + forcast_step):05}.json.gz",
                        )
                    )
                    measurement.append(
                        os.path.join(route_dir, "anno", f"{(seq + idx):05}.json.gz")
                    )

                # # we only store the root and compute the file name when loading,
                # # because storing 40 * long string per sample can go out of memory.
                # measurement.append(os.path.join(route_dir, "anno"))

                # Add the sequence of data to the list
                self.images.append(image)
                self.semantics.append(semantic)
                self.depth.append(depth)
                self.lidars.append(lidar)
                self.boxes.append(box)
                self.future_boxes.append(future_box)
                self.measurements.append(measurement)
                self.sample_start.append(seq)

        del self.angle_distribution
        del self.speed_distribution
        del self.semantic_distribution

        # There is a complex "memory leak"/performance issue when using Python
        # objects like lists in a Dataloader that is loaded with
        # multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects
        # because they only have 1 refcount.
        self.images = np.array(self.images).astype(np.string_)
        self.semantics = np.array(self.semantics).astype(np.string_)
        self.depth = np.array(self.depth).astype(np.string_)
        self.lidars = np.array(self.lidars).astype(np.string_)
        self.boxes = np.array(self.boxes).astype(np.string_)
        self.future_boxes = np.array(self.future_boxes).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)

        self.temporal_lidars = np.array(self.temporal_lidars).astype(np.string_)
        self.temporal_measurements = np.array(self.temporal_measurements).astype(
            np.string_
        )
        self.sample_start = np.array(self.sample_start)
        if rank == 0:
            print(f"Loading {len(self.lidars)} lidars from {len(root)} folders")

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        data = {}

        # Load all image-based data
        data["rgb"] = self._load_image(i)
        data["depth"] = self._load_depth(i)
        data["semantic"] = self._load_semantic(i)
        data["lidar"] = self._load_lidar(i)

        # Load boxes
        bounding_box, future_bounding_box = self._load_boxes(i, data)
        data["bounding_boxes"] = bounding_box
        data["future_bounding_boxes"] = future_bounding_box

        # Load measurements
        measurements = self._load_measurement(i)
        data.update(**measurements)

        return data

    def _load_image(self, index):
        image_file = str(self.images[index][0], encoding="utf-8")
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.config.use_color_aug:
            image = self.image_augmenter_func(image=image)

        image = np.transpose(image, (2, 0, 1))

        return image

    def _load_semantic(self, index):
        semantic_file = str(self.semantics[index][0], encoding="utf-8")
        semantics = cv2.imread(semantic_file, cv2.IMREAD_UNCHANGED)

        semantics = self.converter[semantics]

        semantics = semantics[
            :: self.config.perspective_downsample_factor,
            :: self.config.perspective_downsample_factor,
        ]
        return semantics

    def _load_depth(self, index):
        depth_file = str(self.depth[index][0], encoding="utf-8")
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        cv2.resize(
            depth,
            dsize=(
                depth.shape[1] // self.config.perspective_downsample_factor,
                depth.shape[0] // self.config.perspective_downsample_factor,
            ),
            interpolation=cv2.INTER_LINEAR,
        )
        depth = depth.astype(np.float32) / 255.0

        return depth

    def _load_lidar(self, index):
        lidars_file = str(self.lidars[index][0], encoding="utf-8")
        las_object = laspy.read(lidars_file)
        lidar = las_object.xyz

        lidar_bev = self.lidar_to_histogram_features(
            lidar, use_ground_plane=self.config.use_ground_plane
        )
        lidar_bev = self.lidar_augmenter_func(image=np.transpose(lidar_bev, (1, 2, 0)))
        lidar_bev = np.transpose(lidar_bev, (2, 0, 1))

        return lidar_bev


    def _load_measurement(self, index: int) -> Dict[str, np.ndarray]:
        # Load current measurements
        with gzip.open(self.measurements[index][0], "rt", encoding="utf-8") as f1:
            current_measurement = ujson.load(f1)

        # Gather in a dict so that it has the keys that we expect
        data = {
            "steer": current_measurement["steer"],
            "throttle": current_measurement["throttle"],
            "speed": current_measurement["speed"],
            "theta": current_measurement["theta"],
            "command": t_u.command_to_one_hot(current_measurement["command_near"]),
            "command_far": t_u.command_to_one_hot(current_measurement["command_far"]),
            "next_command": t_u.command_to_one_hot(current_measurement["next_command"]),
        }
        return data


    def _load_boxes(self, index, data):
        # TODO: fix parsing
        # Load measurements file
        with gzip.open(self.boxes[index][0], "rt", encoding="utf-8") as f1:
            current_measurement = ujson.load(f1)
            bounding_boxes = current_measurement["bounding_boxes"]

        with gzip.open(self.future_boxes[index][0], "rt", encoding="utf-8") as f1:
            future_measurement = ujson.load(f1)
            future_bounding_boxes = future_measurement["bounding_boxes"]

        # Translate to expected dict keys
        for bbox in bounding_boxes:
            bbox["position"] = bbox["location"]
            bbox["yaw"] = bbox["rotation"][2]
            del bbox["location"]
        for bbox in future_bounding_boxes:
            bbox["position"] = bbox["location"]
            bbox["yaw"] = bbox["rotation"][2]
            del bbox["location"]

        bounding_boxes, future_bounding_boxes = self._parse_bounding_boxes(
            bounding_boxes,
            future_bounding_boxes,
            y_augmentation=0,  # TODO: get from somewhere
            yaw_augmentation=0,
        )

        bounding_boxes = np.array(bounding_boxes)
        bounding_boxes_padded = np.zeros((self.config.max_num_bbs, 8), dtype=np.float32)

        if bounding_boxes.shape[0] > 0:
          if bounding_boxes.shape[0] <= self.config.max_num_bbs:
            bounding_boxes_padded[:bounding_boxes.shape[0], :] = bounding_boxes
          else:
            bounding_boxes_padded[:self.config.max_num_bbs, :] = bounding_boxes[:self.config.max_num_bbs]

        target_result, avg_factor = \
          self.get_targets(bounding_boxes,
                          self.config.lidar_resolution_height // self.config.bev_down_sample_factor,
                          self.config.lidar_resolution_width // self.config.bev_down_sample_factor)
        data['center_heatmap'] = target_result['center_heatmap_target']
        data['wh'] = target_result['wh_target']
        data['yaw_class'] = target_result['yaw_class_target']
        data['yaw_res'] = target_result['yaw_res_target']
        data['offset'] = target_result['offset_target']
        data['velocity'] = target_result['velocity_target']
        data['brake_target'] = target_result['brake_target']
        data['pixel_weight'] = target_result['pixel_weight']
        data['avg_factor'] = avg_factor

        return bounding_boxes, future_bounding_boxes


    def _parse_bounding_boxes(
        self, boxes, future_boxes=None, y_augmentation=0.0, yaw_augmentation=0
    ):
        if self.config.use_plant and future_boxes is not None:
            # Find ego matrix of the current time step, i.e. the coordinate frame we want to use:
            ego_matrix = None
            ego_yaw = None
            # ego_car always exists
            for ego_candiate in boxes:
                if ego_candiate["class"] == "ego_car":
                    ego_matrix = np.array(ego_candiate["matrix"])
                    ego_yaw = t_u.extract_yaw_from_matrix(ego_matrix)
                    break

        bboxes = []
        future_bboxes = []
        for current_box in boxes:
            # Ego car is always at the origin. We don't predict it.
            if current_box["class"] == "ego_car":
                continue
            # if current_box['class'] in ['weather', 'ego_car', 'landmark', 'ego_info']:
            #   continue
            if "extent" not in current_box.keys():
                continue

            bbox, height = self.get_bbox_label(
                current_box, y_augmentation, yaw_augmentation
            )

            if "num_points" in current_box:
                if (
                    current_box["num_points"]
                    <= self.config.num_lidar_hits_for_detection
                ):
                    continue
            if current_box["class"] == "traffic_light":
                # Only use/detect boxes that are red and affect the ego vehicle
                if not current_box["affects_ego"] or current_box["state"] == "Green":
                    continue

            if current_box["class"] == "stop_sign":
                # Don't detect cleared stop signs.
                if not current_box["affects_ego"]:
                    continue

            # Filter bb that are outside of the LiDAR after the augmentation.
            if (
                bbox[0] <= self.config.min_x
                or bbox[0] >= self.config.max_x
                or bbox[1] <= self.config.min_y
                or bbox[1] >= self.config.max_y
                or height <= self.config.min_z
                or height >= self.config.max_z
            ):  
                continue

            # Load bounding boxes to forcast
            if self.config.use_plant and future_boxes is not None:
                exists = False
                for future_box in future_boxes:
                    # We only forecast boxes visible in the current frame
                    if future_box["id"] == current_box["id"] and future_box[
                        "class"
                    ] in ("car", "walker"):
                        # Found a valid box
                        # Get values in current coordinate system
                        future_box_matrix = np.array(future_box["matrix"])
                        relative_pos = t_u.get_relative_transform(
                            ego_matrix, future_box_matrix
                        )
                        # Update position into current coordinate system
                        future_box["position"] = [
                            relative_pos[0],
                            relative_pos[1],
                            relative_pos[2],
                        ]
                        future_yaw = t_u.extract_yaw_from_matrix(future_box_matrix)
                        relative_yaw = t_u.normalize_angle(future_yaw - ego_yaw)
                        future_box["yaw"] = relative_yaw

                        converted_future_box, _ = self.get_bbox_label(
                            future_box, y_augmentation, yaw_augmentation
                        )
                        quantized_future_box = self.quantize_box(converted_future_box)
                        future_bboxes.append(quantized_future_box)
                        exists = True
                        break

                if not exists:
                    # Bounding box has no future counterpart. Add a dummy with ignore index
                    future_bboxes.append(
                        np.array(
                            [
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                            ]
                        )
                    )

            bbox = t_u.bb_vehicle_to_image_system(
                bbox,
                self.config.pixels_per_meter,
                self.config.min_x,
                self.config.min_y,
            )
            bboxes.append(bbox)
        return bboxes, future_bboxes

    def get_targets(self, gt_bboxes, feat_h, feat_w):
      """
      Compute regression and classification targets in multiple images.

      Args:
          gt_bboxes (list[Tensor]): Ground truth bboxes for each image with shape (num_gts, 4)
            in [tl_x, tl_y, br_x, br_y] format.
          gt_labels (list[Tensor]): class indices corresponding to each box.
          feat_shape (list[int]): feature map shape with value [B, _, H, W]

      Returns:
          tuple[dict,float]: The float value is mean avg_factor, the dict has
            components below:
            - center_heatmap_target (Tensor): targets of center heatmap, shape (B, num_classes, H, W).
            - wh_target (Tensor): targets of wh predict, shape (B, 2, H, W).
            - offset_target (Tensor): targets of offset predict, shape (B, 2, H, W).
            - wh_offset_target_weight (Tensor): weights of wh and offset predict, shape (B, 2, H, W).
          """
      img_h = self.config.lidar_resolution_height
      img_w = self.config.lidar_resolution_width

      width_ratio = float(feat_w / img_w)
      height_ratio = float(feat_h / img_h)

      center_heatmap_target = np.zeros([self.config.num_bb_classes, feat_h, feat_w], dtype=np.float32)
      wh_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
      offset_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
      yaw_class_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
      yaw_res_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
      velocity_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
      brake_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
      pixel_weight = np.zeros([2, feat_h, feat_w], dtype=np.float32)  # 2 is the max of the channels above here.

      if not gt_bboxes.shape[0] > 0:
        target_result = {
            'center_heatmap_target': center_heatmap_target,
            'wh_target': wh_target,
            'yaw_class_target': yaw_class_target.squeeze(0),
            'yaw_res_target': yaw_res_target,
            'offset_target': offset_target,
            'velocity_target': velocity_target,
            'brake_target': brake_target.squeeze(0),
            'pixel_weight': pixel_weight
        }
        return target_result, 1

      center_x = gt_bboxes[:, [0]] * width_ratio
      center_y = gt_bboxes[:, [1]] * height_ratio
      gt_centers = np.concatenate((center_x, center_y), axis=1)

      for j, ct in enumerate(gt_centers):

        ctx_int, cty_int = ct.astype(int)
        ctx, cty = ct
        extent_x = gt_bboxes[j, 2] * width_ratio
        extent_y = gt_bboxes[j, 3] * height_ratio

        radius = g_t.gaussian_radius([extent_y, extent_x], min_overlap=0.1)
        radius = max(2, int(radius))
        ind = gt_bboxes[j, -1].astype(int)

        g_t.gen_gaussian_target(center_heatmap_target[ind], [ctx_int, cty_int], radius)

        wh_target[0, cty_int, ctx_int] = extent_x
        wh_target[1, cty_int, ctx_int] = extent_y

        yaw_class, yaw_res = angle2class(gt_bboxes[j, 4], self.config.num_dir_bins)

        yaw_class_target[0, cty_int, ctx_int] = yaw_class
        yaw_res_target[0, cty_int, ctx_int] = yaw_res

        velocity_target[0, cty_int, ctx_int] = gt_bboxes[j, 5]
        # Brakes can potentially be continous but we classify them now.
        # Using mathematical rounding the split is applied at 0.5
        brake_target[0, cty_int, ctx_int] = int(round(gt_bboxes[j, 6]))

        offset_target[0, cty_int, ctx_int] = ctx - ctx_int
        offset_target[1, cty_int, ctx_int] = cty - cty_int
        # All pixels with a bounding box have a weight of 1 all others have a weight of 0.
        # Used to ignore the pixels without bbs in the loss.
        pixel_weight[:, cty_int, ctx_int] = 1.0

      avg_factor = max(1, np.equal(center_heatmap_target, 1).sum())
      target_result = {
          'center_heatmap_target': center_heatmap_target,
          'wh_target': wh_target,
          'yaw_class_target': yaw_class_target.squeeze(0),
          'yaw_res_target': yaw_res_target,
          'offset_target': offset_target,
          'velocity_target': velocity_target,
          'brake_target': brake_target.squeeze(0),
          'pixel_weight': pixel_weight
      }
      return target_result, avg_factor