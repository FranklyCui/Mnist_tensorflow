# /usr/bin/env python
# -*- conding: -*-

import tensorflow as tf
import matplotlib.pyplot as plt
from download_data import mnist

EPOCHS = 1000

# 待识别图片
X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='x')

# 权重
W = tf.get_variable(name='W', shape=(784, 10), initializer=tf.zeros_initializer)
W_t = tf.get_variable(name='wt', initializer=tf.zeros(shape=(784,10)))

# 偏差
b = tf.get_variable(name='b', initializer=tf.zeros(shape=(10)))

# 输出
print('X.shape:', X.shape)
print('W.shape:', W.shape)
print('b.shape:', b.shape)
y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
print('y_pred.shape: ', y_pred.shape)

# 真实标签
y_real = tf.placeholder(tf.float32, shape=(None, 10))

# 构建交叉墒损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_real * tf.log(y_pred)))

# 创建优化器
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step.minimize(loss=cross_entropy)

# 创建Session, 运行

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Start training...')

    for _ in range(EPOCHS):

