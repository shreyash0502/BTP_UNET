
from PIL import Image
import cv2
import numpy as np
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



def adjust_learning_rate(lr, lr_mode, max_epoch, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    # if lr_mode == 'step':
    #     lr = lr * (0.1 ** (epoch // step))
    if lr_mode == 'poly':
        lr = lr * (1 - epoch / max_epoch) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # print('#'*20)
        # print('learning rate', param_group['lr'])
        # print('#'*20)

    return lr


def get_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            for p in m.parameters():
                yield p