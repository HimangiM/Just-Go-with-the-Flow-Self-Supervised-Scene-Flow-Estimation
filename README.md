# Self-Supervised-Scene-Flow-Estimation

### Installation Requirement
   ```
   CUDA 9.0  
   Tensorflow-gpu 1.9
   Python 3.5
   g++ 5.4.0
   ```
### Compilation of Operation
   The point cloud operations - sampling, grouping, and 3d interpolation can be compiled using make_tf_ops.sh. Check the CUDA compatability and edit the architecture accordingly in Makefiles of each folder (tf_ops/sampling, tf_ops/grouping, tf_ops/3d_interpolation):
   
   ```
   source make_tf_ops.sh
   ```
   
### Datasets
   The data preprocessing files for NuScenes and KITTI are in the src folder: nuscenes_dataset_self_supervised_cycle.py and kitti_dataset_self_supervised_cycle.py. 
  
### Training, Fine-tuning, and Evaluation
   To train the model on nuScenes, execute the script:
   ```
   source src/commands/command_train_1nn_cycle_nuscenes_flip_ons_keep_interp.sh
   ```
   
   To fine tune on KITTI dataset, execute the script:
   ```
   source src/commands/command_train_1nn_cycle_nuscenes_keep_interp_ons_full_fine_tune_kitti_1e4_cache.sh
   ```
   
   
   
    
