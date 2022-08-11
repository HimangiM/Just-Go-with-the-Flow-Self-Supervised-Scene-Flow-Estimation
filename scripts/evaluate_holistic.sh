#!/bin/bash

DATASET_PATH=/mnt/hdd/data/idp/holistic_flow
MODEL_PATH=/home/daniel/projects/idp/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation/models
CODE_PATH=/home/daniel/projects/idp/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation
IMAGE=derkaczda/just-go-with-the-flow:sm_86

docker run -it --rm --gpus all --privileged -v /dev:/dev \
    -v $CODE_PATH:/flow \
    -v $DATASET_PATH:/flow/data_preprocessing/kitti_self_supervised_flow \
    -v $MODEL_PATH:/flow/log_train_pretrained \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    $IMAGE bash src/commands/command_evaluate_holistic.sh