#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: YOLOv3_for_studying 
@File   : parse_config.py
@Author : wanfengcxz
@Date   : 2022/4/18 15:44 
"""


def parse_model_cfg(path):
    """
    Parses the YOLOv3 backbone configuration and returns model definitions.
   :param path: the path of cfg file
   :return: model definitions (list)
   """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.strip() for x in lines]  # remove the blank spaces

    model_defs = []
    for line in lines:
        if line.startswith('['):
            # it represents a new block
            model_defs.append({})
            model_defs[-1]['type'] = line[1:-1].rstrip().lstrip()
            if model_defs[-1]['type'] == 'convolutional':
                model_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            model_defs[-1][key.rstrip()] = value.strip()
    return model_defs


def test_parse_model_cfg():
    model_defs = parse_model_cfg('../cfg/yolov3-tiny.cfg')
    print(model_defs)


if __name__ == '__main__':
    test_parse_model_cfg()
