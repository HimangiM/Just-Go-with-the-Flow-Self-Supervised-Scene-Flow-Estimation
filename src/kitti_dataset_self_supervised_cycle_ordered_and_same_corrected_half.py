'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
import json
import numpy as np
import sys
import pickle
import glob
import random
# import mayavi.mlab as mlab

class SceneflowDataset():
    def __init__(self, root = './data_preprocessing/nuscenes_trainval_pkl',
                 cache_size = 30000, npoints=2048, train=True,
                 softmax_dist = False, num_frames=3, flip_prob=0,
                 sample_start_idx=-1):
        self.npoints = npoints
        self.train = train
        self.root = root
        if self.train:    # 20007
            self.datapath = glob.glob(os.path.join(self.root, 'train/*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'test/*.npz'))
        self.cache = {}
        self.cache_size = cache_size
        self.softmax_dist = softmax_dist
        self.num_frames = num_frames
        self.flip_prob = flip_prob
        self.sample_start_idx = sample_start_idx

    def __getitem__(self, index):
        if index in self.cache:
            pos_list, color_list = self.cache[index]
        else:
            fn = self.datapath[index]
            # pc_list = pickle.load(open(fn, 'rb')) # list of point clouds
            pc_np_list = np.load(fn)
            pc_list = []
            pc_list.append(pc_np_list['pos1'][::2])
            pc_list.append(pc_np_list['pos2'][::2])

            start_idx = np.random.choice(np.arange(len(pc_list)-self.num_frames+1),
                                         size=1)[0]
            pos_list = []
            color_list = []
            min_length = np.min([len(x) for x in pc_list])
            # print (min_length, min_length-self.npoints+1)
            if self.sample_start_idx == -1:
                sample_start_idx = np.random.choice(min_length-self.npoints+1,
                                                    size=1)[0]
            else:
                sample_start_idx = self.sample_start_idx
            sample_idx = np.arange(sample_start_idx,
                                   sample_start_idx+self.npoints)
            for frame_idx in range(start_idx, start_idx + self.num_frames):
                data = pc_list[frame_idx] # num_point x 4
                # sample_idx = np.random.choice(data.shape[0], self.npoints, replace=False)

                pos = data[sample_idx, :3]
                # color = np.tile(data[sample_idx, 3:], [1, 3]) # 2048 x 1 => 2048 x 3
                color = np.zeros((len(sample_idx), 3))

                pos_list.append(pos)
                color_list.append(color)

            prob = random.uniform(0, 1)
            if prob < self.flip_prob:
                pos_list = pos_list[::-1]
                color_list = color_list[::-1]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos_list, color_list)

        return np.array(pos_list), np.array(color_list)

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    # import mayavi.mlab as mlab
    d = SceneflowDataset(npoints=2048, train = False)
    print('Len of dataset:', len(d))
    import time
    tic = time.time()
    for i in range(100):
        # pc1, pc2, c1, c2, flow, m1, m2 = d[i]
        # print (i)
        # pc1, pc2, c1, c2, flow, m1 = d[i]
        pc1, pc2, c1, c2, gt, m1 = d[i]

        # print (pc1.shape)
        # print (pc2.shape)
        # print (c1.shape)
        # print (c2.shape)
        # print (gt.shape)
        # print (m1.shape)
        # print(np.sum(m1))
        # print(np.sum(m2))
        # pc1_m1 = pc1[m1==1,:]
        # pc1_m1_n = pc1[m1==0,:]
        # print(pc1_m1.shape)
        # print(pc1_m1_n.shape)
        # mlab.points3d(pc1_m1[:,0], pc1_m1[:,1], pc1_m1[:,2], scale_factor=0.05, color=(1,0,0))
        # mlab.points3d(pc1_m1_n[:,0], pc1_m1_n[:,1], pc1_m1_n[:,2], scale_factor=0.05, color=(0,1,0))
        # raw_input()

        # mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], scale_factor=0.05, color=(1,0,0))
        # mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], scale_factor=0.05, color=(0,1,0))
        # raw_input()
        # mlab.quiver3d(pc1[:,0], pc1[:,1], pc1[:,2], flow[:,0], flow[:,1], flow[:,2], scale_factor=1)
        # raw_input()

    print(time.time() - tic)
    print(pc1.shape, type(pc1))


