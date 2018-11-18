# /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plt_loss_accuracy(epochs, step, loss, accuracy):
    """
    绘制传入的loss和accuracy
    :param loss: list or tuple, loss value
    :param accuracy: list or tuple, acc
    :return: None
    """
    x = np.arange(start=0, stop=epochs, step=step)
    fig = plt.figure(num='loss & acc')
    ax = fig.add_subplot(111)
    ax.plot(x, loss, label='loss')
    ax.plot(x, accuracy, label='acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('value')
    ax.set_title('loss & acc')
    ax.legend()
    plt.show()
