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
    

