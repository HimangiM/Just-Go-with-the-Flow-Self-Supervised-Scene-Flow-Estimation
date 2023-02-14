#!/bin/bash

DATASET_PATH=/data/datasets/daniel_data_to_move/kitti_sceneflow
MODEL_PATH=/home/danield/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation/models
CODE_PATH=/home/danield/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation
IMAGE=derkaczda/just-go-with-the-flow:sm_75

docker run -it --rm --gpus all --privileged -v /dev:/dev \
    -v $CODE_PATH:/flow \
    -v $DATASET_PATH:/flow/data_preprocessing/kitti_self_supervised_flow \
    -v $MODEL_PATH:/flow/log_train_pretrained \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    $IMAGE bash src/commands/command_evaluate_kitti.sh
