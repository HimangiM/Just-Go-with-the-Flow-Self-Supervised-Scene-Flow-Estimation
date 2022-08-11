import numpy as np
import open3d as o3d
import os
import os.path as osp
import argparse


def create_o3d_pointcloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--sample-id", default=0, type=str)
    args = parser.parse_args()

    # data = np.load(args.data, allow_pickle=True)
    # all_pred, all_points, all_label = data
    id = args.sample_id #str(args.sample_id).zfill(6)
    all_pred = np.load(osp.join(args.data, f"{id}.npz_allpred.npy"), allow_pickle=True)
    all_points = np.load(osp.join(args.data, f"{id}.npz_allpoints.npy"), allow_pickle=True)
    all_label = np.load(osp.join(args.data, f"{id}.npz_alllabel.npy"), allow_pickle=True)

    num_samples = all_pred.shape[0]

    print(f"num_samples={num_samples}")

    pred = all_pred.reshape(-1, all_pred.shape[-1])
    print(pred)
    print(all_points.shape)
    xyz1 = all_points[:, :2048, :3].reshape(-1, 3)
    xyz2 = all_points[:, 2048:, :3].reshape(-1, 3)
    gt_flow = all_label.reshape(-1, 3)

    print(f"shape: pred={pred.shape} xyz1={xyz1.shape} xyz2={xyz2.shape}")

    pcd_frame1 = create_o3d_pointcloud(xyz1, color=[1, 0, 0])
    pcd_frame2 = create_o3d_pointcloud(xyz2, color=[0, 1, 0])
    pcd_flow = create_o3d_pointcloud(xyz1 + pred, color=[0, 0, 1])
    pcd_gt = create_o3d_pointcloud(xyz1 + gt_flow, color=[0, 1, 1])

    print("frame1 is red, frame2 is green and frame1+flow is blue")

    o3d.io.write_point_cloud(osp.join(args.data, f"{id}_frame1.ply"), pcd_frame1)
    o3d.io.write_point_cloud(osp.join(args.data, f"{id}_frame2.ply"), pcd_frame2)
    o3d.io.write_point_cloud(osp.join(args.data, f"{id}_flow.ply"), pcd_flow)
    o3d.io.write_point_cloud(osp.join(args.data, f"{id}_gt.ply"), pcd_gt)
    o3d.visualization.draw_geometries([pcd_frame1, pcd_flow])