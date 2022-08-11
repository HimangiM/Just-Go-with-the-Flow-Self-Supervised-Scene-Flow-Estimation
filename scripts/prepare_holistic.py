import os
import os.path as osp
import numpy as np
import open3d as o3d
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--dest-root", type=str)
    args = parser.parse_args()

    os.makedirs(args.dest_root, exist_ok=True)

    file_suffix = "_mesh.ply"
    files = sorted(glob.glob(osp.join(args.data_root, f"*{file_suffix}")))
    for idx in range(len(files) - 1):
        frame_id = files[idx].split("/")[-1][: -len(file_suffix)]
        print(frame_id)
        mesh1 = o3d.io.read_triangle_mesh(files[idx])
        points1 = np.asarray(mesh1.vertices)
        mesh2 = o3d.io.read_triangle_mesh(files[idx + 1])
        points2 = np.asarray(mesh2.vertices)

        data = {"pos1": points1, "pos2": points2}

        dest = osp.join(args.dest_root, f"{frame_id}.npz")
        with open(dest, "wb") as f:
            np.save(f, data)
