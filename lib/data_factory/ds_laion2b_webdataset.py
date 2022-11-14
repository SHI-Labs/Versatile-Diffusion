import os
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
import torch.distributed as dist
import torchvision.transforms as tvtrans
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
import math
import json
import copy
import pickle
from multiprocessing import shared_memory
import time
from .common import *
from ..log_service import print_log

from lib import visual_service as vis
from .. import sync

import webdataset as wds

###################################################
# this is a special ds that use webdataset mainly #
###################################################

@regdataset()
class laion2b_dummy(ds_base):
    def init_load_info(self):
        self.load_info = []

@regdataset()
class laion2b_webdataset(ds_base):
    def init_load_info(self):
        self.load_info = []

    def make_loader(self, batch_size, num_workers, train=True):
        cfg = self.cfg
        self.root_dir = cfg.root_dir

        interpolation_mode = tvtrans.InterpolationMode.BICUBIC
        if train:
            trans = [
                tvtrans.Resize(cfg.scale, interpolation=interpolation_mode),
                tvtrans.RandomCrop(cfg.scale),
                tvtrans.ToTensor(),]
        else:
            trans = [
                tvtrans.Resize(cfg.scale, interpolation=interpolation_mode),
                tvtrans.CenterCrop(cfg.scale),
                tvtrans.ToTensor(),]

        trans = tvtrans.Compose(trans)

        trans_dict = {'jpg': trans}
        postprocess = customized_postprocess

        shuffle = cfg.get('shuffle', 10000)
        shardshuffle = shuffle > 0
        node_world_size = sync.get_world_size('node')
        nodesplitter = wds.shardlists.split_by_node \
            if node_world_size==1 else wds.shardlists.single_node_only

        tars = [osp.join(self.root_dir, 'data', i) for i in os.listdir(osp.join(self.root_dir, 'data')) 
            if osp.splitext(i)[1]=='.tar']
        tars = sorted(tars)

        dset = wds.WebDataset(
            tars,
            nodesplitter=nodesplitter,
            shardshuffle=shardshuffle,
            handler=wds.warn_and_continue).repeat().shuffle(shuffle)

        print_log(f'Loading webdataset with {len(dset.pipeline[0].urls)} shards.')
        self.min_size = cfg.get('min_size', None)
        self.max_pwatermark = cfg.get('max_pwatermark', None)
        dset = (dset
            .select(self.filter_keys)
            .decode('pil', handler=wds.warn_and_continue)
            .select(self.filter_size)
            .map_dict(**trans_dict, handler=wds.warn_and_continue))

        if postprocess is not None:
            dset = dset.map(postprocess)
 
        dset.batched(batch_size, partial=False)
 
        loader = wds.WebLoader(
            dset, 
            batch_size=None,
            shuffle=False,
            num_workers=num_workers, )
        return loader

    def filter_size(self, x):
        try:
            valid = True
            if self.min_size is not None and self.min_size > 1:
                try:
                    valid = valid and x['json']['original_width'] >= self.min_size and \
                        x['json']['original_height'] >= self.min_size
                except Exception:
                    valid = False
            if self.max_pwatermark is not None and self.max_pwatermark < 1.0:
                try:
                    valid = valid and  x['json']['pwatermark'] <= self.max_pwatermark
                except Exception:
                    valid = False
            return valid
        except Exception:
            return False

    def filter_keys(self, x):
        try:
            return ("jpg" in x) and ("txt" in x)
        except Exception:
            return False

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)

def customized_postprocess(element):
    return element['jpg']*2-1, element['txt'], element['__key__']

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result

###################
# for sd official #
###################

def customized_postprocess_sdofficial(element):
    return {
        'jpg': element['jpg']*2-1, 
        'txt': element['txt'], }

@regdataset()
class laion2b_webdataset_sdofficial(laion2b_webdataset):
    def make_loader(self, batch_size, num_workers, train=True):
        cfg = self.cfg
        self.root_dir = cfg.root_dir

        interpolation_mode = tvtrans.InterpolationMode.BICUBIC
        if train:
            trans = [
                tvtrans.Resize(cfg.scale, interpolation=interpolation_mode),
                tvtrans.RandomCrop(cfg.scale),
                tvtrans.ToTensor(),]
        else:
            trans = [
                tvtrans.Resize(cfg.scale, interpolation=interpolation_mode),
                tvtrans.CenterCrop(cfg.scale),
                tvtrans.ToTensor(),]

        trans = tvtrans.Compose(trans)

        trans_dict = {'jpg': trans}
        postprocess = customized_postprocess_sdofficial

        shuffle = 10000
        shardshuffle = shuffle > 0
        node_world_size = 1
        nodesplitter = wds.shardlists.split_by_node \
            if node_world_size==1 else wds.shardlists.single_node_only

        tars = [osp.join(self.root_dir, 'data', i) for i in os.listdir(osp.join(self.root_dir, 'data')) 
            if osp.splitext(i)[1]=='.tar']
        tars = sorted(tars)

        dset = wds.WebDataset(
            tars,
            nodesplitter=nodesplitter,
            shardshuffle=shardshuffle,
            handler=wds.warn_and_continue).repeat().shuffle(shuffle)

        print(f'Loading webdataset with {len(dset.pipeline[0].urls)} shards.')
        self.min_size = cfg.get('min_size', None)
        self.max_pwatermark = cfg.get('max_pwatermark', None)
        dset = (dset
            .select(self.filter_keys)
            .decode('pil', handler=wds.warn_and_continue)
            .select(self.filter_size)
            .map_dict(**trans_dict, handler=wds.warn_and_continue))

        if postprocess is not None:
            dset = dset.map(postprocess)
 
        dset.batched(batch_size, partial=False, collation_fn=dict_collation_fn)
 
        loader = wds.WebLoader(
            dset, 
            batch_size=None,
            shuffle=False,
            num_workers=num_workers, )
        return loader
