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
    
