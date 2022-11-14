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
class get_transform(object):
    def __init__(self):
        self.transform = {}

    def register(self, transf):
        self.transform[transf.__name__] = transf

    def __call__(self, cfg):
        if cfg is None:
            return None
        if isinstance(cfg, list):
            loader = []
            for ci in cfg:
                t = ci.type
                loader.append(self.transform[t](**ci.args))
            return compose(loader)
        t = cfg.type
        return self.transform[t](**cfg.args)

def register():
    def wrapper(class_):
        get_transform().register(class_)
        return class_
    return wrapper

def have(must=[], may=[]):
    """
    The nextgen decorator that have two list of
        input tells what category the transform
        will operate on. 
    Args:
        must: [] of str,
            the names of the items that must be included
            inside the element. 
            If element[name] exist: do the transform 
            If element[name] is None: raise Exception.
            If element[name] not exist: raise Exception.
        may: [] of str,
            the names of the items that may be contained 
            inside the element for transform. 
            If element[name] exist: do the transform 
            If element[name] is None: ignore it.
            If element[name] not exist: ignore it.
    """
    def route(self, item, e, d):
        """
        Route the element to a proper function
            for calculation.
        Args:
            self: object,
                the transform functor.
            item: str,
                the item name of the data.
            e: {},
                the element
            d: nparray, tensor or PIL.Image,
                the data to transform.
        """
        if isinstance(d, np.ndarray):
            dtype = 'nparray'
        elif isinstance(d, torch.Tensor):
            dtype = 'tensor'
        elif isinstance(d, PIL.Image.Image):
            dtype = 'pilimage'
        else:
            raise ValueError

        # find function by order
        f = None
        for attrname in [
                'exec_{}_{}'.format(item, dtype),
                'exec_{}'.format(item),
                'exec_{}'.format(dtype),
                'exec']:
            f = getattr(self, attrname, None)
            if f is not None:
                break
        d, e = f(d, e)
        e[item] = d
        return e

    def wrapper(func):
        def inner(self, e): 
            e['imsize_previous'] = e['imsize_current']
            imsize_tag_cnt = 0
            imsize_tag = 'imsize_before_' + self.__class__.__name__
            while True:
                if imsize_tag_cnt != 0:
                    tag = imsize_tag + str(imsize_tag_cnt)
                else:
                    tag = imsize_tag
                if not tag in e:
                    e[tag] = e['imsize_current']
                    break
                imsize_tag_cnt += 1
            
            e = func(self, e)
            # must transform list
            for item in must:
                try:
                    d = e[item]
                except:
                    raise ValueError
                if d is None:
                    raise ValueError
                e = route(self, item, e, d)
            # may transform list
            for item in may:
                try:
                    d = e[item]
                except:
                    d = None
                if d is not None:
                    e = route(self, item, e, d)
            return e
        return inner
    return wrapper

class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, element):
        for t in self.transforms:
            element = t(element)
        return element

class TBase(object):
    def __init__(self):
        pass

    def exec(self, data, element):
        raise ValueError

    def rand(self, 
             uid,
             tag, 
             rand_f, 
             *args,
             **kwargs):
        """
        Args:
            uid: string element['unique_id']
            tag: string tells the tag uses when tracking the random number.
                Or the tag to restore the tracked random number.
            rand_f: the random function use to generate random number. 
            **kwargs: the argument for the given random function.
        """
        # if rnduh().hdata is not None:
        #     return rnduh().get_history(uid, self.__class__.__name__, tag)
        # if rnduh().record_path is None:
        #     return rand_f(*args, **kwargs)
        # the special mode to create the random file.
        d = rand_f(*args, **kwargs)
        # rnduh().record(uid, self.__class__.__name__, tag, d)
        return d
