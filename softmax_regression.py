# /usr/bin/env python
# -*- conding: -*-

import tensorflow as tf
import matplotlib.pyplot as plt
from download_data import mnist

EPOCHS = 1000
batch_size = 256
train_loss_lis = []
train_acc_lis = []
test_loss_lis = []
test_acc_lis = []

# 待识别图片
X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='data')

# 权重
W = tf.get_variable(name='weight', shape=(784, 10), initializer=tf.zeros_initializer)

# 偏差
b = tf.get_variable(name='bias', initializer=tf.zeros(shape=(10)))

# 输出预测标签
print('*' * 50)
# print('X.shape:', X.shape)
# print('W.shape:', W.shape)
# print('b.shape:', b.shape)
y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
# print('y_pred.shape: ', y_pred.shape)

# 真实标签
y_real = tf.placeholder(tf.float32, shape=(None, 10), name='y_real')

# 构建交叉墒损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_real * tf.log(y_pred)))

# 判断预测结果准确性  --> 元素为Bool类型
correct_prediction = tf.equal(tf.argmax(y_real, axis=1), tf.argmax(y_pred, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = optimizer.minimize(loss=cross_entropy)

# 创建Session, 运行
with tf.Session() as sess:
    print('Start training...')
    sess.run(tf.global_variables_initializer())
    # 迭代次数, 每次一个batch_size, 而非遍历所有数据!!!
    for epoch in range(EPOCHS):
        # 获取batch数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)
        # 迭代
        sess.run(train_step, feed_dict={X: batch_xs, y_real: batch_ys})
        # sess.run(train_step, feed_dict={X: mnist.train.images, y_real: mnist.train.labels})

        # 输出优化过程
        if epoch % 10 == 0:
            train_loss = sess.run(cross_entropy, feed_dict={X: batch_xs, y_real: batch_ys})
            # train_loss = sess.run(cross_entropy, feed_dict={X: mnist.train.images, y_real: mnist.train.labels})
            test_loss = sess.run(cross_entropy, feed_dict={X: mnist.test.images, y_real: mnist.test.labels})
            accuracy_train = sess.run(accuracy, feed_dict={X: batch_xs, y_real: batch_ys})
            # accuracy_train = sess.run(accuracy, feed_dict={X: mnist.train.images, y_real: mnist.train.labels})
            accuracy_test = sess.run(accuracy, feed_dict={X: mnist.test.images, y_real: mnist.test.labels})

            print(str(epoch)+'th train loss:\t', train_loss)
            print(str(epoch)+'th test loss:\t', test_loss)
            print('train acc: ', accuracy_train)
            print('test acc: ', accuracy_test)

            train_loss_lis.append(train_loss)
            train_acc_lis.append(accuracy_train)
            test_loss_lis.append(test_loss)
            test_acc_lis.append(accuracy_test)

    print('end...')
    print('*' * 50)
    print('loss of train:', sess.run(cross_entropy, feed_dict={X: mnist.train.images, y_real: mnist.train.labels}))
    print('acc of train:', sess.run(accuracy, feed_dict={X: mnist.train.images, y_real: mnist.train.labels}))
    print('loss of test:', sess.run(cross_entropy, feed_dict={X: mnist.test.images, y_real: mnist.test.labels}))
    print('acc of test: ',sess.run(accuracy, feed_dict={X: mnist.test.images, y_real: mnist.test.labels}))

from plot_graph import plt_loss_accuracy
plt_loss_accuracy(epochs=EPOCHS, step=10, loss=train_loss_lis, accuracy=train_acc_lis)