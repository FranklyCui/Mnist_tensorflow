# /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from download_data import mnist
from plot_graph import plt_loss_accuracy


def weight_variable(shape, init=None, name=None):
    """  return variable of weight  """
    if not init:
        init = tf.truncated_normal(shape)
    return tf.Variable(init, name=name)

def bias_variable(shape, init=None, name=None):
    """  return variable of bias  """
    if not init:
        init = tf.ones(shape=(32,))
    return tf.Variable(init, name=name)

def conv(img, w):
    """ return conv result of img and w  """
    return tf.nn.conv2d(input=img, filter=w, strides=(1,1,1,1), padding='SAME')

def max_pool(img, ksize=(1,2,2,1), padding='SAME'):
    # ksize: A list or tuple of 4 ints. The size of the window for each dimension of the input tensor
    return tf.nn.max_pool(img, ksize=ksize, padding=padding)

if __name__ == '__main__':

    batch_size = 128

    # 读取数据
    x = tf.placeholder(tf.float32, shape=(None, 784))
    y_pred = tf.placeholder(tf.float32, shape=(None, 10))

    # 处理输入数据: [batch, in_height, in_width, in_channels]
    x_imag = tf.reshape(x, shape=(-1, 28, 28, 1))

    # 第一层卷积: [filter_height, filter_width, in_channels, out_channels]
    w_conv1 = weight_variable(shape=(5, 5, 1, 32), name='w_conv1')
    b_conv1 = bias_variable(shape=(32,), name='b_conv1')
    out_conv1 = conv(img=x_imag, w=w_conv1) + b_conv1
    act_conv1 = tf.nn.relu(out_conv1)
    pool_conv1 = max_pool(act_conv1)

    # 第二层卷积
    w_conv2 = weight_variable(shape=(5, 5, 32, 64), name='w_conv2')
    b_conv2 = bias_variable(shape=(64,), name='b_conv2')
    out_conv2 = conv(img=pool_conv1, w=w_conv2) + b_conv2
    act_conv2 = tf.nn.relu(out_conv2)
    pool_conv2 = max_pool(img=act_conv2)

    # 第一层全连接: channels=1024
    w_fc1 = weight_variable(shape=(7*7*64, 1024), name='w_fc1')
    b_fc1 = bias_variable(shape=(1024,), name='b_fc1')
    pool_conv2_flatten = tf.reshape(pool_conv2, shape=(-1, 7*7*64))
    out_fc1 = tf.matmul(pool_conv2_flatten, w_fc1) + b_fc1
    act_fc1 = tf.nn.relu(out_fc1)

    # Dropout层
    keep_prob = tf.placeholder(tf.float32)
    drop_fc1 = tf.nn.dropout(act_fc1, keep_prob=keep_prob)

    # 第二层全连接: classes = channels=10
    w_fc2 = weight_variable(shape=(1024, 10))
    b_fc2 = bias_variable(shape=(10,))
    





