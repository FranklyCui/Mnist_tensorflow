# /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image
import matplotlib.pyplot as plt
from download_data import mnist

def save_pic(save_dir=None, num=100):
    """
    读取数据集并保存为图片格式
    :param save_dir: 保存路径
    :param num: 读取图片数量
    :return: None
    """
    # 图片路径
    if save_dir:
        save_dir = os.path.join('MNIST_data', 'raw')

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # 保存前20张
    for i in range(num):
        image_array = mnist.train.images[i]
        # print('image_array: ', image_array.shape, type(image_array))
        imag = image_array.reshape((28, 28))
        # print('image_array.shape:', imag.shape)
        plt.axis('off')
        # 文件名
        filename = os.path.join(save_dir, 'mnist_train_%d.jpg' % i)
        plt.imshow(imag)
        plt.savefig(filename)

    print('figs have saved in: %s.' % save_dir)