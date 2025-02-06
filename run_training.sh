#!/bin/bash

# Activate anaconda environment
module load Anaconda3/2024.02-1
module load libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

conda activate garage
conda env list

export CARLA_ROOT=/cluster/work/andrebw/carla_garage/carla/CARLA_Leaderboard_20
export WORK_DIR=/cluster/work/andrebw/carla_garage_main
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

cd $WORK_DIR
pwd

# Epochs: 31
torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=1 --rdzv_id=42353467 --rdzv_backend=c10d \
/cluster/work/andrebw/carla_garage_main/team_code/train.py \
  --id tfpp-debug \
  --epochs 31 \
  --batch_size 8 \
  --use_focal_loss 0 \
  --logdir logs \
  --use_disk_cache 0 \
  --cpu_cores 24 \
  --augment 0 \
  --use_plant 0 \
  --num_repetitions 1 \
  --learn_multi_task_weights 0 \
  --use_wp_gru 1 \
  --use_controller_input_prediction 1 \
  --use_discrete_command 1 \
  --use_bev_semantic 1 \
  --use_semantic 1 \
  --detect_boxes 1 \
  --use_depth 1 \
  --setting "all" \
  --root_dir /cluster/work/andrebw/carla_garage/database/dataset_default_expert2024_11_08/data
  # --root_dir /cluster/work/andrebw/carla_garage/database/dataset_4hz_2024_10_26/data 
  # --root_dir /cluster/work/andrebw/carla_garage_main/dataset/hb_dataset_v01_2024_10_17
  # --load_file /cluster/work/andrebw/carla_garage_main/logs/transfuser++\
#  --local_rank 0
#   --lr 0.001 \
  #--load_file \  # LOAD_FILE
                          # Model to load for initialization.Expects the full path
                          # with ending /path/to/model.pth Optimizer files are
                          # expected to exist in the same directory
  # --setting \  # SETTING     What training setting to use. Options: all: Train on
                          # all towns no validation data. 01_03_withheld: Do not
                          # train on Town 01 and Town 03. 02_05_withheld: Do not
                          # train on Town 02 and Town 05. 04_06_withheld: Do not
                          # train on Town 04 and Town 06. Withheld data is used
                          # for validation
  #--schedule_reduce_epoch_01 \  # SCHEDULE_REDUCE_EPOCH_01
                          # Epoch at which to reduce the lr by a factor of 10 the
                          # first time. Only used with --schedule 1
  #--schedule_reduce_epoch_02 \  # SCHEDULE_REDUCE_EPOCH_02
                          # Epoch at which to reduce the lr by a factor of 10 the
                          # second time. Only used with --schedule >
  #--backbone \  # BACKBONE   Which fusion backbone to use. Options: transFuser,
                          # aim, bev_encoder
  #--image_architecture \  # IMAGE_ARCHITECTURE
                          # Which architecture to use for the image branch.
                          # resnet34, regnety_032 etc. All options of the TIMM lib
                          # can be used but some might need adjustments to the
                          # backbone.
  #--lidar_architecture \  # LIDAR_ARCHITECTURE
                          # Which architecture to use for the lidar branch.
                          # Tested: resnet34, regnety_032.Has the special video
                          # option video_resnet18 and video_swin_tiny.
  #--use_velocity 1 \  # USE_VELOCITY
                          # Whether to use the velocity input. Expected values are
                          # 0:False, 1:True
  #--n_layer \  # N_LAYER     Number of transformer layers used in the transfuser
  #--val_every \  # VAL_EVERY
                          # At which epoch frequency to validate.
  #--sync_batch_norm \  # SYNC_BATCH_NORM
                          # 0: Compute batch norm for each GPU independently, 1:
                          # Synchronize batch norms across GPUs.
  #--zero_redundancy_optimizer \  # ZERO_REDUNDANCY_OPTIMIZER
                          # 0: Normal AdamW Optimizer, 1: Use zero-redundancy
                          # Optimizer to reduce memory footprint.
  #--use_disk_cache \  # USE_DISK_CACHE
                          # 0: Do not cache the dataset 1: Cache the dataset on
                          # the disk pointed to by the SCRATCH environment
                          # variable. Useful if the dataset is stored on shared
                          # slow filesystem and can be temporarily stored on
                          # faster SSD storage on the compute node.
  #--lidar_seq_len \  # LIDAR_SEQ_LEN
                          # How many temporal frames in the LiDAR to use. 1 equals
                          # single timestep.
  #--realign_lidar \  # REALIGN_LIDAR
                          # Whether to realign the temporal LiDAR frames, to all
                          # lie in the same coordinate frame.
  #--use_ground_plane \  # USE_GROUND_PLANE
                          # Whether to use the ground plane of the LiDAR. Only
                          # affects methods using the LiDAR.
  #--pred_len \  # PRED_LEN   Number of waypoints the model predicts
  #--estimate_class_distributions \  # ESTIMATE_CLASS_DISTRIBUTIONS
                          # Whether to estimate the weights to re-balance CE
                          # loss, or use the config default.
  #--use_focal_loss \  # USE_FOCAL_LOSS
                          # Whether to use focal loss instead of cross entropy
                          # for target speed classification.
  #--use_cosine_schedule \  # USE_COSINE_SCHEDULE
                          # Whether to use a cyclic cosine learning rate schedule
                          # instead of the linear one.
  #--augment \  # AUGMENT     # Whether to use rotation and translation augmentation
  #--use_plant \  # USE_PLANT
                          # If true trains a privileged PlanT model, otherwise a
                          # sensorimotor agent like TF++
  #--learn_origin \  # LEARN_ORIGIN
                          # Whether to learn the origin of the waypoints or use
                          # 0/0
  #--local_rank \  # LOCAL_RANK
                          # Local rank for launch with torch.launch. Default =
                          # -999 means not used.
  #--train_sampling_rate \  # TRAIN_SAMPLING_RATE
                          # Rate at which the dataset is sub-sampled during
                          # training.Should be an odd number ideally ending with 1
                          # or 5, because of the LiDAR sweeps alternating every
                          # frame
  #--use_amp \  # USE_AMP     Currently amp produces inf gradients. DO NOT
                          # USE!.Whether to use automatic mixed precision with
                          # fp16 during training.
  #--use_grad_clip \  # USE_GRAD_CLIP
                          # Whether to clip the gradients during training.
  #--use_color_aug \  # USE_COLOR_AUG
                          # Whether to use color augmentation on the images.
  #--use_semantic \  # USE_SEMANTIC
                          # Whether to use semantic segmentation as auxiliary loss
  #--use_depth \  # USE_DEPTH
                          # Whether to use depth prediction as auxiliary loss for
                          # training.
  #--detect_boxes \  # DETECT_BOXES
                          # Whether to use the bounding box auxiliary task.
  #--use_bev_semantic \  # USE_BEV_SEMANTIC
                          # Whether to use bev semantic segmentation as auxiliary
                          # loss for training.
  #--estimate_semantic_distribution \  # ESTIMATE_SEMANTIC_DISTRIBUTION
                          # Whether to estimate the weights to rebalance the
                          # semantic segmentation loss by class.This is extremely
                          # slow.
  #--gru_hidden_size \  # GRU_HIDDEN_SIZE
                          # Number of features used in the hidden size of the GRUs
  #--use_cutout \  # USE_CUTOUT
                          # Whether to use the cutout data augmentation technique.
  #--add_features \  # ADD_FEATURES
                          # Whether to add (or concatenate) the features at the
                          # end of the backbone.
  #--freeze_backbone \  # FREEZE_BACKBONE
                          # Freezes the encoder and auxiliary heads. Should be
                          # used when loading a already trained model. Can be used
                          # for fine-tuning or multi-stage training.
  #--learn_multi_task_weights \  # LEARN_MULTI_TASK_WEIGHTS
                          # Whether to learn the multi-task weights according to
                          # https://arxiv.org/abs/1705.07115.
  #--transformer_decoder_join \  # TRANSFORMER_DECODER_JOIN
                          # Whether to use a transformer decoder instead of global
                          # average pool + MLP for planning.
  #--bev_down_sample_factor \  # BEV_DOWN_SAMPLE_FACTOR
                          # Factor (int) by which the bev auxiliary tasks are
                          # down-sampled.
  #--perspective_downsample_factor \  # PERSPECTIVE_DOWNSAMPLE_FACTOR
                          # Factor (int) by which the perspective auxiliary tasks
                          # are down-sampled.
  #--gru_input_size \  # GRU_INPUT_SIZE
                          # Number of channels in the InterFuser GRU input and
                          # Transformer decoder.Must be divisible by number of
                          # heads (8)
  #--bev_grid_height_downsample_factor \  # BEV_GRID_HEIGHT_DOWNSAMPLE_FACTOR
                          # Ratio by which the height size of the voxel grid in
                          # BEV decoder are larger than width and depth. Value
                          # should be >= 1. Larger values uses less gpu memory.
                          # Only relevant for the bev_encoder backbone.
  #--wp_dilation \  # WP_DILATION
                          # Factor by which the wp are dilated compared to full
                          # CARLA 20 FPS
  #--continue_epoch \  # CONTINUE_EPOCH
                          # Whether to continue the training from the loaded epoch
                          # or from 0.
  #--max_height_lidar \  # MAX_HEIGHT_LIDAR
                          # Points higher than this threshold are removed from the
                          # LiDAR.
  #--smooth_route \  # SMOOTH_ROUTE
                          # Whether to smooth the route points with linear
                          # interpolation.
  #--num_lidar_hits_for_detection \  # NUM_LIDAR_HITS_FOR_DETECTION
                          # Number of LiDAR hits a bounding box needs to have in
                          # order to be used.
  #--use_speed_weights \  # USE_SPEED_WEIGHTS
                          # Whether to weight target speed classes.
  #--max_num_bbs \  # MAX_NUM_BBS
                          # Maximum number of bounding boxes our system can
                          # detect.
  #--use_optim_groups \  # USE_OPTIM_GROUPS
                          # Whether to use optimizer groups to exclude some
                          # parameters from weight decay
  #--weight_decay \  # WEIGHT_DECAY
                          # Weight decay coefficient used during training
  #--use_plant_labels \  # USE_PLANT_LABELS
                          # Whether to use the relabeling from plant or the
                          # original labels.Does not work with focal loss because
                          # the implementation does not support soft targets.
  #--use_label_smoothing \  # USE_LABEL_SMOOTHING
                          # Whether to use label smoothing in the classification
                          # losses. Not working as intended when combined with
                          # use_speed_weights.
  #--tp_attention \  # TP_ATTENTION
                          # Adds a TP at the TF decoder and computes it with
                          # attention visualization. Only compatible with
                          # transformer decoder.
  #--multi_wp_output \  # MULTI_WP_OUTPUT
                          # Predict 2 WP outputs and select between them. Only
                          # compatible with use_wp=1, transformer_decoder_join=1
