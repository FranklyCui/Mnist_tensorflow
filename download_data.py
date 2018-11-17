#/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载mnist数据集, 并返回

"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(train_dir='MNIST_data/', one_hot=True)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('train.image.shape:', mnist.train.images.shape)
    images = mnist.train.images[0:9]
    print('images.shape:',images.shape)
    images = images.reshape(-1, 28, 28)
    print('images.shape:', images.shape)
    fig = plt.figure('mnist')
    # axes = fig.add_subplot(3, 3, 4)
    # axes.imshow(images[3])
    # plt.show()

    for row in range(3):
        for col in range(3):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1)
            ax.imshow(images[row * 3 + col])
    plt.show()

    print('train label: ', mnist.train.labels[:3])