# /usr/bin/env python
# -*- conding:utf-8 -*-

import numpy as np
from download_data import mnist

for i in range(20):
    one_hot_label = mnist.train.labels[i, :]
    if i == 0:
        print('lable[0]:', one_hot_label[0])
    label = np.argmax(one_hot_label, axis=0)
    print('mnist_train_%d.jpg\tlabel:%d' % (i, label))



