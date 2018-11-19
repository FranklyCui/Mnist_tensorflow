# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
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
        init = tf.ones(shape=shape)
    return tf.Variable(init, name=name)

def conv(img, w):
    """ return conv result of img and w  """
    return tf.nn.conv2d(input=img, filter=w, strides=(1,1,1,1), padding='SAME')

def max_pool(img, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME'):
    # ksize: A list or tuple of 4 ints. The size of the window for each dimension of the input tensor
    return tf.nn.max_pool(img, ksize=ksize, strides=strides, padding=padding)

if __name__ == '__main__':

    batch_size = 128
    info = 10
    EPOCHS = 3000
    KEEP_PROB = 0.9
    train_loss_lis = []
    train_acc_lis = []
    valid_loss_lis = []
    valid_acc_lis = []
    filename = 'data_for_plot.csv'

    # 读取数据
    x = tf.placeholder(tf.float32, shape=(None, 784))
    y_real = tf.placeholder(tf.float32, shape=(None, 10))

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
    print('b_conv2:', b_conv2)
    print('conv: ', conv(img=pool_conv1, w=w_conv2))
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
    y_pred = tf.matmul(drop_fc1, w_fc2) + b_fc2

    # 直接采用softmax_cross_entropy计算交叉熵, 而非先计算softmax, 再计算cross_entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_real, logits=y_pred))

    # 定义优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_step = optimizer.minimize(loss=cross_entropy)

    # 定义准确率
    correct_predication = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y_real, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predication, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)

            # 每100步打印验证集准确度
            if epoch % info == 0:
                train_loss = cross_entropy.eval(feed_dict={x: batch_xs, y_real: batch_ys, keep_prob: 1.0})
                train_acc = accuracy.eval(feed_dict={x: batch_xs, y_real: batch_ys, keep_prob: 1.0})       # 测试和验证一般不用dropout
                batch_xs_val, batch_ys_val = mnist.validation.next_batch(batch_size=batch_size)
                valid_loss = cross_entropy.eval(feed_dict={x: batch_xs_val, y_real: batch_ys_val, keep_prob: 1.0})
                valid_acc = accuracy.eval(feed_dict={x: batch_xs_val, y_real: batch_ys_val, keep_prob: 1.0})
                print('%d epoch:\ttrain_loss: %.3f\ttrain_acc: %0.3f\tvalid_acc: %0.3f' % (epoch, train_loss, train_acc, valid_acc))

                train_loss_lis.append(train_loss)
                train_acc_lis.append(train_acc)
                valid_loss_lis.append(valid_loss)
                valid_acc_lis.append(valid_acc)

                # print('y_pred_batch_xs: ', sess.run(y_pred, feed_dict={x: batch_xs, keep_prob: 1.0}).shape)
                # print('y_real: ', sess.run(y_real, feed_dict={y_real: batch_ys}).shape)
                # print('batch_xs.shape: ', batch_xs.shape)
                # print('batch_ys.shape: ', batch_ys.shape)

            train_step.run(feed_dict={x: batch_xs, y_real: batch_ys, keep_prob: KEEP_PROB})

        test_loss = cross_entropy.eval(feed_dict={x: mnist.test.images[:1000], y_real: mnist.test.labels[:1000], keep_prob: 1.0})
        test_acc = accuracy.eval(feed_dict={x: mnist.test.images[:1000], y_real: mnist.test.labels[:1000], keep_prob: 1.0})
        print('test_loss: ', test_loss)
        print('test_acc: ', test_acc)

        plt_data = pd.DataFrame(data={'train_loss': train_loss_lis, 'train_acc': train_acc_lis,
                           'valid_loss': valid_loss_lis, 'valid_acc': valid_acc_lis})
        plt_data.to_csv(filename)

        print('y_pred: ', sess.run(y_pred, feed_dict={x: mnist.test.images[:1000], keep_prob: 1.0}).shape)
        print('y_real: ', sess.run(y_real, feed_dict={y_real: mnist.test.labels[:1000]}).shape)


        # test: batch_xs 的作用域: 不局限于for循环内
        # print('batch_xs.shape: ', batch_xs.shape)

        plt_data = pd.read_csv(filename)
        plt_loss_accuracy(epochs=EPOCHS, step=10, loss=plt_data['train_loss'], accuracy=plt_data['valid_acc'])



