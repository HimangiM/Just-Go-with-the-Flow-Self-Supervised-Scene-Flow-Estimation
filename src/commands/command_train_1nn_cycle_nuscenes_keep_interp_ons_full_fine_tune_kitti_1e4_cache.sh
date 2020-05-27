


python src/train_cycle_fine_tune_kitti.py \
    --model model_concat_upsa_cycle \
    --data data_preprocessing/kitti_self_supervised_flow \
    --log_dir log_train_cycle_fine_tune_kitti \
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
    --kitti_dataset data_preprocessing/kitti_self_supervised_flow \
    --sample_start_idx 0
