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
from tf_grouping import query_ball_point, group_point, knn_point


def placeholder_inputs(batch_size, num_point, num_frames=3):
    # change here, num_point*2 -> numpoint*5
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point * num_frames, 6))
    # labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    # masks_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    # return pointclouds_pl, labels_pl, masks_pl
    return pointclouds_pl


def get_model(radius, layer, point_cloud, is_training, bn_decay=None, knn=False,
              flow_module='default', num_frames=2, stop_gradient=False,
              rigidity=False, rigidity_radius=0.5, rigidity_nsample=4, rgb=False):

    num_point = point_cloud.get_shape()[1].value // num_frames

    pred_flow, end_points = get_model_flow(radius, layer, point_cloud, is_training,
                                      bn_decay=None, knn=False,
                                      flow_module='default')

    pred_f = pred_flow + point_cloud[:, :num_point, :3] # flow + p1 = pred_f => pc2_hat

    _, idx = knn_point(1, point_cloud[:, num_point:num_point*2, :3],
                       pred_f)

    print ('Knn idx shape:', idx.shape)
    grouped_xyz = group_point(point_cloud[:, num_point:num_point*2, :3], idx)
    print ('Grouped xyz shape:', grouped_xyz.shape)

    grouped_xyz = tf.squeeze(grouped_xyz, axis=2) # grouped_xyz => pc2nn
    end_points_f = {
        'idx': idx,
        'pred_flow': pred_flow,
        'pc2': point_cloud[:, num_point:num_point*2, :3]
    }

    if rigidity:
        pc1 = point_cloud[:, :2048, :3]
        rigid_idx, _ = query_ball_point(rigidity_radius, rigidity_nsample, pc1,
                                        pc1)
        rigid_grouped_flow = group_point(pred_flow, rigid_idx)
        end_points_f['rigid_group_flow'] = rigid_grouped_flow
        end_points_f['rigid_pc1_flow'] = pred_flow

    if rgb:
        pred_f_rgb, dist_f, grouped_xyz_rgb_f = get_interpolated_rgb(pred_f, point_cloud[:, num_point:])
        end_points_f['pred_f_rgb'] = pred_f_rgb
        end_points_f['dist_f'] = dist_f
        end_points_f['grouped_xyz_rgb_f'] = grouped_xyz_rgb_f


    # changes from here
    if stop_gradient:
        print ('Stop gradient on predicted point cloud 2')
        pred_f_copy = tf.Variable(0, dtype=pred_f.dtype, trainable=False, collections=[])
        pred_f_copy = tf.assign(pred_f_copy, pred_f, validate_shape=False)
    else:
        pred_f_copy = pred_f

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):

        pred_fb_xyz = (pred_f_copy + grouped_xyz) / 2

        pred_fb = tf.concat([pred_fb_xyz, point_cloud[:, :num_point, 3:]], axis = 2)

        # num_point = pred_f (predicted point cloud 2), num_point:num_point*2 = point cloud 1
        point_cloud_back = tf.concat([pred_fb, point_cloud[:, :num_point]], axis = 1)

        # import ipdb; ipdb.set_trace()

        pred_flow_back, end_points = get_model_flow(radius, layer, point_cloud_back, is_training,
                                                    bn_decay=None, knn=False,
                                                    flow_module='default')

        pred_b = pred_flow_back + pred_fb_xyz

        end_points_b = {
            'pred_flow_b': pred_flow_back,
        }

        if rgb:
            pred_b_rgb, dist_b, grouped_xyz_rgb_b = get_interpolated_rgb(pred_b, point_cloud[:, :num_point])
            end_points_f['pred_b_rgb'] = pred_b_rgb
            end_points_f['dist_b'] = dist_b
            end_points_f['grouped_xyz_rgb_b'] = grouped_xyz_rgb_b

    return pred_f, pred_b, grouped_xyz, end_points_f, end_points_b

def get_model_flow(radius, layer, point_cloud, is_training, bn_decay=None, knn=False, flow_module='default'):
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


def get_loss(pred, label):
    """ pred: BxNx3,
        label: BxNx3,
        mask: BxN
    """
    batch_size = pred.get_shape()[0].value
    num_point = pred.get_shape()[1].value
    l2_loss = tf.reduce_mean(
        tf.reduce_sum((pred - label) * (pred - label), axis=2) / 2.0)
    tf.summary.scalar('l2 loss', l2_loss)
    tf.add_to_collection('losses', l2_loss)
    return l2_loss


def get_cycle_loss(pred_f, grouped_xyz, pred_b, point_cloud1, end_points=None,
                   rigidity=False, rgb=False, point_cloud1_rgb=None, flip_prefix='', cycle_loss_weight=1,
                   knn_loss_weight=1):

    end_points_loss = {}

    knn_l2_loss = knn_loss_weight*tf.reduce_mean(
        tf.reduce_sum((pred_f - grouped_xyz) * (pred_f - grouped_xyz), axis=2) / 2.0)
    tf.summary.scalar('{}KNN L2 loss'.format(flip_prefix), knn_l2_loss)
    tf.add_to_collection('{}KNN losses'.format(flip_prefix), knn_l2_loss)

    end_points_loss['knn_l2_loss'] = knn_l2_loss

    cycle_l2_loss = cycle_loss_weight*tf.reduce_mean(
        tf.reduce_sum((pred_b - point_cloud1) * (pred_b - point_cloud1), axis=2) / 2.0)
    tf.summary.scalar('{}Cycle l2 loss'.format(flip_prefix), cycle_l2_loss)
    tf.add_to_collection('{}Cycle losses'.format(flip_prefix), cycle_l2_loss)

    end_points_loss['cycle_l2_loss'] = cycle_l2_loss

    l2_loss = knn_l2_loss + cycle_l2_loss

    avg_distance_metric = tf.reduce_mean(
        tf.reduce_sum((pred_f - grouped_xyz) * (pred_f - grouped_xyz), axis=2) ** 0.5)
    tf.summary.scalar('{}Avg Distance Metric loss'.format(flip_prefix), avg_distance_metric)
    tf.add_to_collection('{}Avg Distance Metric losses'.format(flip_prefix), avg_distance_metric)

    if rigidity:
        rigid_group_flow = end_points['rigid_group_flow']
        rigid_pc1_flow = tf.expand_dims(end_points['rigid_pc1_flow'], 2)

        rigidity_loss = tf.reduce_mean(
        tf.reduce_sum((rigid_group_flow - rigid_pc1_flow) * (rigid_group_flow - rigid_pc1_flow),
                      axis=3) / 2.0)
        tf.summary.scalar('{}Rigidity loss'.format(flip_prefix), rigidity_loss)
        tf.add_to_collection('{}Rigidity losses'.format(flip_prefix), rigidity_loss)

        end_points_loss['rigidity_loss'] = rigidity_loss

        l2_loss = l2_loss + rigidity_loss

    if rgb:
        pred_f_rgb = end_points['pred_f_rgb']
        rgb_loss_f = 10*tf.reduce_mean(
        tf.reduce_sum((pred_f_rgb - point_cloud1_rgb) * (pred_f_rgb - point_cloud1_rgb), axis=2) / 2.0)

        end_points_loss['rgb_loss_f'] = rgb_loss_f

        pred_b_rgb = end_points['pred_b_rgb']
        rgb_loss_b = 10*tf.reduce_mean(
        tf.reduce_sum((pred_b_rgb - point_cloud1_rgb) * (pred_b_rgb - point_cloud1_rgb), axis=2) / 2.0)

        end_points_loss['rgb_loss_b'] = rgb_loss_b

        rgb_loss = rgb_loss_f + rgb_loss_b
        tf.summary.scalar('{}RGB Loss Forward'.format(flip_prefix), rgb_loss_f)
        tf.add_to_collection('{}RGB Loss Forward'.format(flip_prefix), rgb_loss_f)

        tf.summary.scalar('{}RGB Loss Backward'.format(flip_prefix), rgb_loss_b)
        tf.add_to_collection('{}RGB Loss Backward'.format(flip_prefix), rgb_loss_b)

        tf.summary.scalar('{}RGB Loss'.format(flip_prefix), rgb_loss)
        tf.add_to_collection('{}RGB Loss'.format(flip_prefix), rgb_loss)

        end_points_loss['rgb_loss'] = rgb_loss
        l2_loss = l2_loss + rgb_loss

    end_points_loss['l2_loss'] = l2_loss
    tf.summary.scalar('{}Total l2 loss'.format(flip_prefix), l2_loss)
    tf.add_to_collection('{}Total losses'.format(flip_prefix), l2_loss)

    return l2_loss, end_points_loss

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024 * 2, 6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
