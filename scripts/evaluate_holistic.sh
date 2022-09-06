#!/bin/bash

DATASET_PATH=/data/datasets/holistic_flow/train #/mnt/hdd/data/idp/holistic_flow
MODEL_PATH=/backup/otoc/flow/expr/4955d4b50ef69b1026d933a954afe372
CODE_PATH=/home/danield/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation
IMAGE=derkaczda/just-go-with-the-flow:sm_75

docker run -it --rm --gpus \"device=0\" \
    -v $CODE_PATH:/flow \
    -v $DATASET_PATH:/flow/data_preprocessing/kitti_self_supervised_flow \
    -v $MODEL_PATH:/flow/log_train_pretrained \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    $IMAGE bash src/commands/command_evaluate_holistic.sh
