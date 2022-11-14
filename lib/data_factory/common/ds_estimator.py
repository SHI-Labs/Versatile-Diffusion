import os.path as osp
import numpy as np
import numpy.random as npr
import PIL
import cv2

import torch
import torchvision
import xml.etree.ElementTree as ET
import json
import copy
import math

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_estimator(object):
    def __init__(self):
        self.estimator = {}

    def register(self, estimf):
        self.estimator[estimf.__name__] = estimf

    def __call__(self, cfg):
        if cfg is None:
            return None
        t = cfg.type
        return self.estimator[t](**cfg.args)

def register():
    def wrapper(class_):
        get_estimator().register(class_)
        return class_
    return wrapper

@register()
class PickFileEstimator(object):
    """
    This is an estimator that filter load_info
        using the provided filelist
    """
    def __init__(self, 
                 filelist = None,
                 repeat_n = 1):
        """
        Args:
            filelist: a list of string gives the name of images 
                we would like to visualize, evaluate or train. 
            repeat_n: int, times these images will be repeated
        """
        self.filelist = filelist
        self.repeat_n = repeat_n

    def __call__(self, load_info):
        load_info_new = []
        for info in load_info:
            if os.path.basename(info['image_path']).split('.')[0] in self.filelist:
                load_info_new.append(info)
        return load_info_new * self.repeat_n

@register()
class PickIndexEstimator(object):
    """
    This is an estimator that filter load_info
        using the provided indices
    """
    def __init__(self, 
                 indexlist = None,
                 **kwargs):
        """
        Args:
            indexlist: [] of int.
                the indices to be filtered out. 
        """
        self.indexlist = indexlist

    def __call__(self, load_info):
        load_info_new = [load_info[i] for i in self.indexlist]
        return load_info_new
