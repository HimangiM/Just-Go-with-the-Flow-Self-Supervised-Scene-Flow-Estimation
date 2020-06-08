"""
    FlowNet3D model with up convolution
"""

import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import utils.tf_util
from utils.pointnet_util import *


def placeholder_inputs(batch_size, num_point):
    # change here, num_point*2 -> numpoint*5
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point * 2, 6))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    masks_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, masks_pl


def get_model(radius, layer, point_cloud, is_training, bn_decay=None, knn=False, flow_module='default'):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    print ('knn value: ', knn)

    end_points = {}
    batch_size = point_cloud.get_shape()[0].value  # batch_size = 16
    num_point = point_cloud.get_shape()[1].value // 2
    # change here, num_point hard coded to 2048
    # num_point = 2048

    l0_xyz_f1 = point_cloud[:, :num_point, 0:3]
    l0_points_f1 = point_cloud[:, :num_point, 3:]
    l0_xyz_f2 = point_cloud[:, num_point:, 0:3]
    l0_points_f2 = point_cloud[:, num_point:, 3:]

    RADIUS1 = 0.5
    RADIUS2 = 1.0
    RADIUS3 = 2.0
    RADIUS4 = 4.0

    with tf.variable_scope('sa1') as scope:
        # radius, npoints, nlayers, mlp size, sampling technique
        # Set conv layers, POINT FEATURE LEARNING
        # Frame 1, Layer 1 (with radius = 0.5)
        l1_xyz_f1, l1_points_f1, l1_indices_f1 = pointnet_sa_module(l0_xyz_f1,
                                                                    l0_points_f1,
                                                                    npoint=1024,
                                                                    radius=RADIUS1,
                                                                    nsample=16,
                                                                    mlp=[32, 32,
                                                                         64],
                                                                    mlp2=None,
                                                                    group_all=False,
                                                                    is_training=is_training,
                                                                    bn_decay=bn_decay,
                                                                    scope='layer1',
                                                                    knn=knn)
        end_points['l1_indices_f1'] = l1_indices_f1
        end_points['l1_xyz_f1'] = l1_points_f1
        end_points['l1_input_f1'] = l0_xyz_f1

        # Frame 1, Layer 2 (with radius = 1.0), Inputs are the above function's output
        l2_xyz_f1, l2_points_f1, l2_indices_f1 = pointnet_sa_module(l1_xyz_f1,
                                                                    l1_points_f1,
                                                                    npoint=256,
                                                                    radius=RADIUS2,
                                                                    nsample=16,
                                                                    mlp=[64, 64,
                                                                         128],
                                                                    mlp2=None,
                                                                    group_all=False,
                                                                    is_training=is_training,
                                                                    bn_decay=bn_decay,
                                                                    scope='layer2',
                                                                    knn=knn)
        end_points['l2_indices_f1'] = l2_indices_f1
        end_points['l2_xyz_f1'] = l2_points_f1
        end_points['l2_input_f1'] = l1_xyz_f1

        scope.reuse_variables()
        # Frame 2, Layer 1 (with radius = 0.5)
        l1_xyz_f2, l1_points_f2, l1_indices_f2 = pointnet_sa_module(l0_xyz_f2,
                                                                    l0_points_f2,
                                                                    npoint=1024,
                                                                    radius=RADIUS1,
                                                                    nsample=16,
                                                                    mlp=[32, 32,
                                                                         64],
                                                                    mlp2=None,
                                                                    group_all=False,
                                                                    is_training=is_training,
                                                                    bn_decay=bn_decay,
                                                                    scope='layer1',
                                                                    knn=knn)
        end_points['l1_points_f2'] = l1_points_f2
        end_points['l1_xyz_f2'] = l1_indices_f2
        end_points['l1_input_f2'] = l0_xyz_f2
        # Tensor("sa1/layer1_1/GatherPoint:0", shape=(16, 1024, 3), dtype=float32, device= / device: GPU:0)
        # Tensor("sa1/layer1_1/Squeeze:0", shape=(16, 1024, 64), dtype=float32, device= / device: GPU:0)
        # Tensor("sa1/layer1_1/QueryBallPoint:0", shape=(16, 1024, 16), dtype=int32, device= / device: GPU:0)


        # Frame 2, Layer 2(with radius = 1.0), input are of the above function's output
        l2_xyz_f2, l2_points_f2, l2_indices_f2 = pointnet_sa_module(l1_xyz_f2,
                                                                    l1_points_f2,
                                                                    npoint=256,
                                                                    radius=RADIUS2,
                                                                    nsample=16,
                                                                    mlp=[64, 64,
                                                                         128],
                                                                    mlp2=None,
                                                                    group_all=False,
                                                                    is_training=is_training,
                                                                    bn_decay=bn_decay,
                                                                    scope='layer2',
                                                                    knn=knn)
        end_points['l2_points_f2'] = l2_points_f2
        end_points['l2_xyz_f2'] = l2_indices_f2
        end_points['l2_input_f2'] = l1_xyz_f2


        # Tensor("sa1/layer2_1/GatherPoint:0", shape=(16, 256, 3), dtype=float32, device= / device: GPU:0)
        # Tensor("sa1/layer2_1/Squeeze:0", shape=(16, 256, 128), dtype=float32, device= / device: GPU:0)
        # Tensor("sa1/layer2_1/QueryBallPoint:0", shape=(16, 256, 16), dtype=int32, device= / device: GPU:0)

    # POINT MIXTURE
    # embedding layer
    # radius = 1, 10, 50
    print("Radius here:", radius)
    print('KNN', knn)
    print('flow module', flow_module)
    if flow_module == 'default':
        _, l2_points_f1_new = flow_embedding_module(l2_xyz_f1, l2_xyz_f2,
                                                    l2_points_f1, l2_points_f2,
                                                    radius=radius, nsample=64,
                                                    mlp=[128, 128, 128],
                                                    is_training=is_training,
                                                    bn_decay=bn_decay,
                                                    scope='flow_embedding', bn=True,
                                                    pooling='max', knn=True,
                                                    corr_func='concat')
        end_points['l2_points_f1_new'] = l2_points_f1_new
    elif flow_module == 'all':
        _, l2_points_f1_new = flow_embedding_module_all(l2_xyz_f1, l2_xyz_f2,
                                                    l2_points_f1, l2_points_f2,
                                                    radius=radius, nsample=256,
                                                    mlp=[128, 128, 128],
                                                    is_training=is_training,
                                                    bn_decay=bn_decay,
                                                    scope='flow_embedding', bn=True,
                                                    pooling='max', knn=True,
                                                    corr_func='concat')
        end_points['l2_points_f1_new'] = l2_points_f1_new

    # setconv layer
    # Layer 3 with radius = 2.0
    l3_xyz_f1, l3_points_f1, l3_indices_f1 = pointnet_sa_module(l2_xyz_f1,
                                                                l2_points_f1_new,
                                                                npoint=64,
                                                                radius=RADIUS3,
                                                                nsample=8,
                                                                mlp=[128, 128,
                                                                     256],
                                                                mlp2=None,
                                                                group_all=False,
                                                                is_training=is_training,
                                                                bn_decay=bn_decay,
                                                                scope='layer3')
    end_points['l3_indices_f1'] = l3_indices_f1
    end_points['l3_xyz_f1'] = l3_points_f1
    # Tensor("layer3/GatherPoint:0", shape=(16, 64, 3), dtype=float32, device=/device:GPU:0)
    # Tensor("layer3/Squeeze:0", shape=(16, 64, 256), dtype=float32, device=/device:GPU:0)
    # Tensor("layer3/QueryBallPoint:0", shape=(16, 64, 8), dtype=int32, device=/device:GPU:0)

    # Layer 4 with radius = 4.0
    l4_xyz_f1, l4_points_f1, l4_indices_f1 = pointnet_sa_module(l3_xyz_f1,
                                                                l3_points_f1,
                                                                npoint=16,
                                                                radius=RADIUS4,
                                                                nsample=8,
                                                                mlp=[256, 256,
                                                                     512],
                                                                mlp2=None,
                                                                group_all=False,
                                                                is_training=is_training,
                                                                bn_decay=bn_decay,
                                                                scope='layer4')
    end_points['l4_indices_f1'] = l4_indices_f1
    end_points['l4_xyz_f1'] = l4_points_f1
    # Tensor("layer4/GatherPoint:0", shape=(16, 16, 3), dtype=float32, device=/device:GPU:0)
    # Tensor("layer4/Squeeze:0", shape=(16, 16, 512), dtype=float32, device=/device:GPU:0)
    # Tensor("layer4/QueryBallPoint:0", shape=(16, 16, 8), dtype=int32, device=/device:GPU:0)

    ### FLOW REFINEMENT MODULE
    # Feature Propagation
    # Frame 1, l1->l2; l2->l3; l3->l4
    l3_feat_f1 = set_upconv_module(l3_xyz_f1, l4_xyz_f1, l3_points_f1,
                                   l4_points_f1, nsample=8, radius=2.4, mlp=[],
                                   mlp2=[256, 256], scope='up_sa_layer1',
                                   is_training=is_training, bn_decay=bn_decay,
                                   knn=True)
    end_points['l3_feat_f1'] = l3_feat_f1

    l2_feat_f1 = set_upconv_module(l2_xyz_f1, l3_xyz_f1, tf.concat(axis=-1,
                                                                   values=[
                                                                       l2_points_f1,
                                                                       l2_points_f1_new]),
                                   l3_feat_f1, nsample=8, radius=1.2,
                                   mlp=[128, 128, 256], mlp2=[256],
                                   scope='up_sa_layer2',
                                   is_training=is_training, bn_decay=bn_decay,
                                   knn=True)
    end_points['l2_feat_f1'] = l2_feat_f1

    l1_feat_f1 = set_upconv_module(l1_xyz_f1, l2_xyz_f1, l1_points_f1,
                                   l2_feat_f1, nsample=8, radius=0.6,
                                   mlp=[128, 128, 256], mlp2=[256],
                                   scope='up_sa_layer3',
                                   is_training=is_training, bn_decay=bn_decay,
                                   knn=True)
    end_points['l1_feat_f1'] = l1_feat_f1

    if layer == 'pointnet':
        l0_feat_f1 = pointnet_fp_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1,
                                        l1_feat_f1, [256, 256], is_training,
                                        bn_decay, scope='fa_layer4')
    else:
        print ('Last set conv layer running')
        l0_feat_f1 = set_upconv_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1,
                                       l1_feat_f1, nsample=8, radius=0.3,
                                       mlp=[128,128,256], mlp2=[256],
                                       scope='up_sa_layer4',
                                       is_training=is_training, bn_decay=bn_decay,
                                       knn=True)
    end_points['l0_feat_f1'] = l0_feat_f1

    # FC layers
    net = tf_util.conv1d(l0_feat_f1, 128, 1, padding='VALID', bn=True,
                         is_training=is_training, scope='fc1',
                         bn_decay=bn_decay)

    end_points['net1'] = net
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None,
                         scope='fc2')

    end_points['net'] = net
    return net, end_points


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(losses)


def get_loss(pred, label, mask, end_points):
    """ pred: BxNx3,
        label: BxNx3,
        mask: BxN
    """
    batch_size = pred.get_shape()[0].value
    num_point = pred.get_shape()[1].value
    l2_loss = tf.reduce_mean(
        mask * tf.reduce_sum((pred - label) * (pred - label), axis=2) / 2.0)
    tf.summary.scalar('l2 loss', l2_loss)
    tf.add_to_collection('losses', l2_loss)
    return l2_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024 * 2, 6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
