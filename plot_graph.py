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

    ax1 = fig.add_subplot(111)
    ax1.plot(x, loss, label='loss', c='r')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_xlim((0, epochs))
    ax1.set_title('Loss and acc')
    ax1.legend(loc=1)

    ax2 = ax1.twinx()
    ax2.plot(x, accuracy, label='acc', c='b')
    ax2.set_ylim((0, 1))
    ax2.set_ylabel('Accuracy Rate')
    ax2.legend(loc=2)

    plt.show()
    print('end')


if __name__ == '__main__':

    print('start...')

    x = np.arange(start=0, stop=100, step=5)
    y1 = np.power(x, 3)
    print('y1:', y1)
    y2 = np.arange(0, 1, step=5/100)
    print('y2: ', y2)

    plt_loss_accuracy(epochs=100, step=5, loss=y1, accuracy=y2)


