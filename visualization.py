import numpy as np
import open3d as o3d
import os

_dir = 'kitti_self_supervised_flow/train'
file_name = '000000.npz'
point_cloud = np.load(os.path.join(_dir, file_name))

pc1 = point_cloud['pos1']
pc2 = point_cloud['pos2']
gt_flow = point_cloud['gt'] + pc1

print ('Point Cloud 1 shape:{}, Point Cloud 2 shape:{}, Flow Shape:{}'.format(pc1.shape, pc2.shape, gt_flow.shape))

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(pc1)
pcd1.paint_uniform_color([1,0,0])

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pc2)
pcd2.paint_uniform_color([0,1,0])

pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(gt_flow)
pcd3.paint_uniform_color([0,0,1])

o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
