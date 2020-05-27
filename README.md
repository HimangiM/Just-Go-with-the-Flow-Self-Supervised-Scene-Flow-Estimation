# Just Go with the Flow: Self-Supervised Scene Flow Estimation

Code release for the paper Just Go with the Flow: Self-Supervised Scene Flow Estimation, CVPR 2020 (Oral).

Authors: Himangi Mittal, Brian Okorn, David Held

[arxiv] [Project Page]

### Introduction
In this work, we propose a method of scene flow estimation using two self-supervised losses, based on nearest neighbors and cycle consistency. These self-supervised losses allow us to train our method on large unlabeled autonomous driving datasets; the resulting method matches current state-of-the-art supervised performance using no real world annotations and exceeds stateof-the-art performance when combining our self-supervised approach with supervised learning on a smaller labeled dataset.

For more details, please refer to our paper or project page.

### Installation 
#### Requirements
   ```
   CUDA 9.0  
   Tensorflow-gpu 1.9
   Python 3.5
   g++ 5.4.0
   ```
#### Steps
  (a). Clone the repository.
  ```
  git clone https://github.com/HimangiM/Self-Supervised-Scene-Flow-Estimation.git
  ```
  (b). Install dependencies
  ```
  Create a virtualenv
  python3 -m venv sceneflowvenv
  source sceneflowvenv/bin/activate
  cd Self-Supervised-Scene-Flow-Estimation
  pip install -r requirements.txt
  ```
  ```
  Check for CUDA-9.0
  ```
  (c). Compile the operations
  
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
   
   
   
    
