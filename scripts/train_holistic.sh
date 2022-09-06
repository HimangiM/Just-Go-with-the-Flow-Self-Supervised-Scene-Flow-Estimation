#!/bin/bash

DATASET_PATH=/data/datasets/otoc/flow/datasets/dd06550c9437020ced0d19e3a8caf01b
MODEL_PATH=/home/danield/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation/models/9fc7e63d579c8ac584291ae74ec531a8
CODE_PATH=/home/danield/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation
LOG_PATH=/backup/otoc/flow/expr/adf7c3188e9349655de3c59ad09ae0ee
IMAGE=derkaczda/just-go-with-the-flow:sm_75

docker run -it --rm --gpus all --privileged -v /dev:/dev \
    -v $CODE_PATH:/flow \
    -v $DATASET_PATH:/flow/data_preprocessing/kitti_self_supervised_flow \
    -v $MODEL_PATH:/flow/log_train_pretrained \
    -v $LOG_PATH:/flow/log_train_holistic \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    $IMAGE bash src/commands/command_train_cycle_fine_tune_holistic.sh
