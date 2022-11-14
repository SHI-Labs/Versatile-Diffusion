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

from ...cfg_holder import cfg_unique_holder as cfguh

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_loader(object):
    def __init__(self):
        self.loader = {}

    def register(self, loadf):
        self.loader[loadf.__name__] = loadf

    def __call__(self, cfg):
        if cfg is None:
            return None
        if isinstance(cfg, list):
            loader = []
            for ci in cfg:
                t = ci.type
                loader.append(self.loader[t](**ci.args))
            return compose(loader)
        t = cfg.type
        return self.loader[t](**cfg.args)

class compose(object):
    def __init__(self, loaders):
        self.loaders = loaders

    def __call__(self, element):
        for l in self.loaders:
            element = l(element)
        return element
    
    def __getitem__(self, idx):
        return self.loaders[idx]

def register():
    def wrapper(class_):
        get_loader().register(class_)
        return class_
    return wrapper

def pre_loader_checkings(ltype):
    lpath = ltype+'_path'
    # cache feature added on 20201021
    lcache = ltype+'_cache'
    def wrapper(func):
        def inner(self, element):
            if lcache in element:
                # cache feature added on 20201021
                data = element[lcache]
            else:
                if ltype in element:
                    raise ValueError
                if lpath not in element:
                    raise ValueError

                if element[lpath] is None:
                    data = None
                else:
                    data = func(self, element[lpath], element)
            element[ltype] = data

            if ltype == 'image':
                if isinstance(data, np.ndarray):
                    imsize = data.shape[-2:]
                elif isinstance(data, PIL.Image.Image):
                    imsize = data.size[::-1]
                elif isinstance(data, torch.Tensor):
                    imsize = [data.size(-2), data.size(-1)]
                elif data is None:
                    imsize = None
                else:
                    raise ValueError
                element['imsize'] = imsize
                element['imsize_current'] = copy.deepcopy(imsize)
            return element
        return inner
    return wrapper
