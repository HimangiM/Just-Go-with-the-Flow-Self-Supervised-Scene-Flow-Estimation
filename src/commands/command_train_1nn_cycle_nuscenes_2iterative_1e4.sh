
# this file is for retraining of iterative method till 2 iterations, that is
# pc2_hat, pc2_double_hat, pc2_triple_hat

python src/train_1nn_cycle_nuscenes.py \
    --model model_concat_upsa_1nn_cycle_nuscenes_iterative \
    --data data_preprocessing/nuscenes_trainval_rgb_pkl_600_full \
    --log_dir log_train_1nn_cycle_nuscenes_2iterative_1e4 \
    --num_point 2048 \
    --batch_size 2 \
    --radius 5 \
    --layer pointnet \
    --cache_size 0 \
    --gpu 2 \
    --learning_rate 0.0001 \
    --dataset nuscenes_dataset_self_supervised_cycle_ordered_and_same_corrected_rgb \
    --num_frames 2 \
    --fine_tune \
    --max_epoch 10000 \
    --model_path pretrained_models/log_train_pretrained/model.ckpt \
    --flip_prob 0.5
#    \
#    > log_evaluate.txt 2>&1 &
