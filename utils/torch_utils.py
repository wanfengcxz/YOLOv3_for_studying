#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: YOLOv3_for_studying 
@File   : torch_utils.py
@Author : wanfengcxz
@Date   : 2022/4/18 21:57 
"""

import torch


def init_seeds(seed=0):
    """
    Sets the random seed, so that we can reappear the result of experiment ago.
    :param seed: random seed
    """
    torch.manual_seed(seed)  # Sets random seed for a cpu
    torch.cuda.manual_seed(seed)  # Sets random seed for a gpu
    torch.cuda.manual_seed_all(seed)  # Sets random seed for all gpus


def select_device(force_cpu=False):
    """
    Selects the gpu or cpu for running.
    :param force_cpu:
    :return:
    """

    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if not cuda:
        print('Using cpu')
    if cuda:
        c = 1024 ** 2  # Converts B to MB (1MB=1024*1024B)
        num_gpu = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(num_gpu)]
        print("Using CUDA device0 _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
              (x[0].name, x[0].total_memory / c))
        if num_gpu > 1:
            for i in range(1, num_gpu):
                print("           device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                      (i, x[i].name, x[i].total_memory / c))
    return device


if __name__ == '__main__':
    select_device()