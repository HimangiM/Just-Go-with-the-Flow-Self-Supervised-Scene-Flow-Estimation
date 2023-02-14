#!/bin/bash

DATASET_PATH=/data/datasets/holistic_flow
MODEL_PATH=/home/danield/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation/flownet3d_pretrained_model
CODE_PATH=/home/danield/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation
LOG_PATH=$CODE_PATH/log_train_holistic_flownet3d
IMAGE=derkaczda/just-go-with-the-flow:sm_75

docker run -it --rm --gpus \"device=1\" \
    -v $CODE_PATH:/flow \
    -v $DATASET_PATH:/flow/data_preprocessing/kitti_self_supervised_flow \
    -v $MODEL_PATH:/flow/log_train_pretrained \
    -v $LOG_PATH:/flow/log_train_holistic \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    $IMAGE bash src/commands/command_train_cycle_fine_tune_holistic.sh
