


python src/train_1nn_cycle_nuscenes_fine_tune_kitti.py \
    --model model_concat_upsa_1nn_cycle_nuscenes \
    --data data_preprocessing/kitti_rm_ground_cycle \
    --log_dir log_train_1nn_cycle_nuscenes_keep_interp_ons_fine_tune_kitti_1e4_cache_p2nn_10k \
    --num_point 2048 \
    --batch_size 8 \
    --radius 5 \
    --layer pointnet \
    --cache_size 30000 \
    --gpu 2 \
    --learning_rate 0.0001 \
    --dataset kitti_dataset_self_supervised_cycle \
    --num_frames 2 \
    --max_epoch 10000 \
    --fine_tune \
    --model_path log_train_pretrained/model.ckpt \
    --kitti_dataset data_preprocessing/kitti_rm_ground_cycle \
    --sample_start_idx 0
#    \
#    > log_evaluate.txt 2>&1 &
