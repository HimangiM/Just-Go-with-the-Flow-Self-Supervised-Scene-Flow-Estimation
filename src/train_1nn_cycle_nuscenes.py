
import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
# import ipdb
from tempfile import TemporaryFile
from tensorflow.python import debug as tf_debug

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pickle

# arguments start from here

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa', help='Model name [default: model_concat_upsa]')
parser.add_argument('--data', default='data_preprocessing/data_processed_maxcut_35_20k_2k_8192', help='Dataset directory [default: data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
parser.add_argument('--log_dir', default='log_train', help='Log dir [default: log_train]')
parser.add_argument('--model_path', default='log_train_pretrained/model.ckpt', help='model weights path')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=150, help='Epoch to run [default: 151]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--radius', type=float, default=5.0, help='Radius of flow embedding layer')
parser.add_argument('--layer', type=str, default='pointnet', help='Last layer for upconv')
parser.add_argument('--flow', type=str, default='default', help='flow embedding module type')
parser.add_argument('--cache_size', type=int, default=30000, help='knn or query ball point')
parser.add_argument('--softmax', action='store_true', help='softmax in sampling')
parser.add_argument('--knn', action='store_true', help='knn or query ball point')
parser.add_argument('--numfiles', type=int, default=100, help='Number of files to fine tune on')
parser.add_argument('--num_frames', type=int, default=3, help='Number of frames to run cycle')
parser.add_argument('--fine_tune', action='store_true', help='load trained model and resume batch')
parser.add_argument('--dataset', type=str, default='flying_things_dataset', help='dataset to train')
parser.add_argument('--stop_gradient', action='store_true', help='Stop gradient for predicted point cloud 2')
parser.add_argument('--flip_prob', type=float, default=0, help='Probability to flip the point cloud frames')
parser.add_argument('--rigidity', action='store_true', help='Rigidity')
parser.add_argument('--rgb', action='store_true', help='RGB')
parser.add_argument('--cycle_loss_weight', type=float, default=1, help='Weight for cycle loss')
parser.add_argument('--knn_loss_weight', type=float, default=1, help='Weight for KNN loss')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

EPOCH_CNT = 0

KNN_LOSS_WEIGHT = FLAGS.knn_loss_weight
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATA = FLAGS.data
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
RADIUS = FLAGS.radius
LAYER = FLAGS.layer
FLOW_MODULE = FLAGS.flow
CACHE_SIZE = FLAGS.cache_size
KNN = FLAGS.knn
SOFTMAX_ARG = FLAGS.softmax
NUM_FILES = FLAGS.numfiles
NUM_FRAMES = FLAGS.num_frames
FINE_TUNE = FLAGS.fine_tune
STOP_GRADIENT = FLAGS.stop_gradient
FLIP_PROB=FLAGS.flip_prob
RIGIDITY = FLAGS.rigidity
RGB = FLAGS.rgb
CYCLE_LOSS_WEIGHT = FLAGS.cycle_loss_weight

print(FLAGS)

DATASET = importlib.import_module(FLAGS.dataset)

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model + '.py')
MODEL_PATH = FLAGS.model_path
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
TRAIN_DATASET = DATASET.SceneflowDataset(DATA, npoints=NUM_POINT,
                                         cache_size=CACHE_SIZE, softmax_dist=SOFTMAX_ARG,
                                         train=True, num_frames = NUM_FRAMES, flip_prob=FLIP_PROB)
print ('len of train: ', len(TRAIN_DATASET))

os.system('cp %s %s' % (__file__, LOG_DIR))  # bkp of train procedure
os.system('cp %s %s' % ('{}.py'.format(FLAGS.dataset), LOG_DIR))  # bkp of dataset file
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TEST_DATASET = DATASET.SceneflowDataset(DATA, npoints=NUM_POINT, train=False, num_frames = NUM_FRAMES)
print ('len of test: ', len(TEST_DATASET))

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.000001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            # pointclouds_pl = [16, 4096, 6], labels_pl = [16, 2048, 3], masks_pl = [16, 2048]
            pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAMES)
            # ipdb.set_trace()
            # a = tf.slice(pointclouds_pl, [1, 0, 0], [1, 1, 6])
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)  # batch = 0
            bn_decay = get_bn_decay(batch)    # bn_decay = 0.5
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            pred_f, pred_b, label_nn, end_points_f, end_points_b = MODEL.get_model(RADIUS,
                                                                                      LAYER,
                                                                                      pointclouds_pl,
                                                                                      is_training_pl,
                                                                                      bn_decay=bn_decay,
                                                                                      knn=KNN,
                                                                                      flow_module=FLOW_MODULE,
                                                                                      num_frames=NUM_FRAMES,
                                                                                      stop_gradient=STOP_GRADIENT,
                                                                                      rigidity=RIGIDITY,
                                                                                      rgb=RGB)

            loss, end_points_loss = MODEL.get_cycle_loss(pred_f = pred_f, grouped_xyz = label_nn,
                                        pred_b = pred_b,
                                        point_cloud1 = pointclouds_pl[:, :NUM_POINT, :3],
                                        end_points=end_points_f, rigidity=RIGIDITY,
                                        rgb=RGB, point_cloud1_rgb=pointclouds_pl[:, :NUM_POINT, 3:],
                                        cycle_loss_weight=CYCLE_LOSS_WEIGHT)        ### L2 Loss
            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)        ### 0.001 in the arguments
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':       ### given in the arguments
                optimizer = tf.train.AdamOptimizer(learning_rate)
            # two step below
            # train_op = optimizer.minimize(loss, global_step=batch)

            grad_var = optimizer.compute_gradients(loss)
            # grads = tf.gradients(loss, tf.trainable_variables())
            # grad_var = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grad_var, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root='/media/gaurav/DATADRIVE0/himangi/tf_dbg')

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        if FINE_TUNE:
            print ('fine tuning, model path:', MODEL_PATH)
            saver.restore(sess, MODEL_PATH)
            log_string('Pretrained model restored')
            # ipdb.set_trace()
            init_new_vars_op = tf.initialize_variables([batch])
            sess.run(init_new_vars_op)
        else:
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'label': label_nn,
               'is_training_pl': is_training_pl,
               'pred': pred_f,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'grad_var': grad_var,
               'end_points_loss': end_points_loss,
               'end_points_f': end_points_f}

        eval_one_epoch(sess, ops, test_writer)
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            print("PROGRESS: {}%".format(((epoch+1) / MAX_EPOCH) * 100))

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    # change here,  numpoint *(5, 3)
    batch_data = np.zeros((bsize, NUM_POINT * 2, 6))
    batch_label = np.zeros((bsize, NUM_POINT, 3))
    batch_mask = np.zeros((bsize, NUM_POINT))
    # shuffle idx to change point order (change FPS behavior)
    shuffle_idx = np.arange(NUM_POINT)
    # change here
    shuffle_idx2 = np.arange(NUM_POINT)
    np.random.shuffle(shuffle_idx)
    np.random.shuffle(shuffle_idx2)

    for i in range(bsize):
        # ipdb.set_trace()
        # if dataset[0] == None:
        #     print (i, bsize)
        # import ipdb; ipdb.set_trace()
        pc1, pc2, color1, color2, flow, mask1 = dataset[idxs[i + start_idx]]

        # move pc1 to center
        pc1_center = np.mean(pc1, 0)
        pc1 -= pc1_center
        pc2 -= pc1_center
        batch_data[i, :NUM_POINT, :3] = pc1[shuffle_idx]
        batch_data[i, :NUM_POINT, 3:] = color1[shuffle_idx]
        batch_data[i, NUM_POINT:, :3] = pc2[shuffle_idx2]

        batch_data[i, NUM_POINT:, 3:] = color2[shuffle_idx2]
        batch_label[i] = flow[shuffle_idx]
        batch_mask[i] = mask1[shuffle_idx]

    return batch_data, batch_label, batch_mask


def get_cycle_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    # change here,  numpoint *(5, 3)
    batch_data = np.zeros((bsize, NUM_POINT * NUM_FRAMES, 6))

    shuffle_idx = np.arange(NUM_POINT)

    for i in range(bsize):
        # ipdb.set_trace()
        # if dataset[0] == None:
        #     print (i, bsize)
        pos, color = dataset[idxs[i + start_idx]]

        pos1_center = np.mean(pos[0], 0) # 1 * 3

        for frame_idx in range(NUM_FRAMES):
            np.random.shuffle(shuffle_idx)
            batch_data[i, NUM_POINT*frame_idx:NUM_POINT*(frame_idx+1), :3] = \
                pos[frame_idx, shuffle_idx, :] - pos1_center
            batch_data[i, NUM_POINT*frame_idx:NUM_POINT*(frame_idx+1), 3:] = \
                color[frame_idx, shuffle_idx, :]

    return batch_data

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    outfile = TemporaryFile()

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    print ('length here:', len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) // BATCH_SIZE
    log_string('Len of dataset: %f' % len(TRAIN_DATASET))
    log_string(str(datetime.now()))

    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data = get_cycle_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['is_training_pl']: is_training, }
        summary, step, _, grad_var_val, \
        loss_val, pred_val, label_val, end_points_loss_val, \
        end_points_f = sess.run([ops['merged'], ops['step'],
                                                  ops['train_op'],
                                                  ops['grad_var'][1:],
                                                  ops['loss'],
                                                  ops['pred'],
                                                  ops['label'],
                                                  ops['end_points_loss'],
                                                  ops['end_points_f']], feed_dict=feed_dict)
        # print('Train end points loss val losses', end_points_loss_val)
        for g, v in grad_var_val:
            if np.isnan(g).any():
                print('gradient is nan')
                ipdb.set_trace()
            if np.isnan(v).any():
                print('variable is nan')
                ipdb.set_trace()
        if np.isnan(loss_val):
            print('>>>>>> NAN <<<<<<<<')
            ipdb.set_trace()

        # x = np.arange(16)
        # np.save('pointcloud', pred_val)
        # print ('point cloud value here:', ops['pointclouds_pl'])

        ### OPTIC FLOW HERE
        # print ('pred_val: ', pred_val.shape, type(pred_val))

        train_writer.add_summary(summary, step)
        loss_sum += loss_val

        if (batch_idx + 1) % 1 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
            log_string('Cycle Train mean loss: %f' % (loss_sum / 2))
            log_string('Cycle Train all losses {}'.format(end_points_loss_val))
            loss_sum = 0

def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    test_idxs = np.arange(0, len(TEST_DATASET))
    print ('length here:', len(TEST_DATASET))
    # np.random.shuffle(train_idxs)
    num_batches = len(TEST_DATASET) // BATCH_SIZE
    log_string('Len of dataset: %f' % len(TEST_DATASET))
    log_string(str(datetime.now()))

    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data = get_cycle_batch(TEST_DATASET, test_idxs,
                                     start_idx, end_idx)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['is_training_pl']: is_training, }
        summary, step, \
        loss_val, pred_val, label_val, end_points_loss_val, end_points_f = sess.run([ops['merged'], ops['step'],
                                                                    ops['loss'],
                                                                    ops['pred'],
                                                                    ops['label'],
                                                                    ops['end_points_loss'],
                                                                    ops['end_points_f']],
                                                                    feed_dict=feed_dict)


        loss_sum += loss_val
        log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
        log_string('loss: %f' % (loss_val))
        log_string('Eval all losses {}'.format(end_points_loss_val))
        # ipdb.set_trace()


    EPOCH_CNT += 1
    avg_loss = loss_sum / float(len(TEST_DATASET) / BATCH_SIZE)
    summary = tf.Summary(value=[tf.Summary.Value(tag="loss",
                                                 simple_value=avg_loss)])
    test_writer.add_summary(summary, step)
    log_string('avg loss: %f' % (avg_loss))
    return loss_sum / float(len(TEST_DATASET) / BATCH_SIZE)


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
