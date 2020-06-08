#!/bin/bash
#
#SBATCH --job-name=kitti_evaluate
#SBATCH --output=/home/hmittal/github-codes/Self-Supervised-Scene-Flow-Estimation/output_eval.txt
#SBATCH --error=/home/hmittal/github-codes/Self-Supervised-Scene-Flow-Estimation/error_eval.txt
#
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=10000
#SBATCH --gres=gpu:1


python src/evaluate_cycle_kitti.py \
    --model model_concat_upsa \
    --data data_preprocessing/kitti_self_supervised_flow \
    --model_path log_train_pretrained/model.ckpt \
    --kitti_dataset data_preprocessing/kitti_self_supervised_flow \
    --num_point 2048 \
    --batch_size 8 \
    --radius 5 \
    --layer pointnet \
    --gpu 2 \
    --num_frames 2
    

