import os
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
import torch.distributed as dist
import torchvision
import copy
import itertools

from ... import sync
from ...cfg_holder import cfg_unique_holder as cfguh
from ...log_service import print_log

import torch.distributed as dist
from multiprocessing import shared_memory

# import multiprocessing
# if hasattr(multiprocessing, "shared_memory"):
#     from multiprocessing import shared_memory
# else:
#     # workaround for single gpu inference on colab
#     shared_memory = None
    
import pickle
import hashlib
import random

class ds_base(torch.utils.data.Dataset):
    def __init__(self, 
                 cfg, 
                 loader = None, 
                 estimator = None, 
                 transforms = None, 
                 formatter = None):

        self.cfg = cfg
        self.load_info = None
        self.init_load_info()
        self.loader = loader
        self.transforms = transforms
        self.formatter = formatter

        if self.load_info is not None:
            load_info_order_by = getattr(self.cfg, 'load_info_order_by', 'default')
            if load_info_order_by == 'default':
                self.load_info = sorted(self.load_info, key=lambda x:x['unique_id'])
            else: 
                try:
                    load_info_order_by, reverse = load_info_order_by.split('|')
                    reverse = reverse == 'reverse'
                except:
                    reverse = False
                self.load_info = sorted(
                    self.load_info, key=lambda x:x[load_info_order_by], reverse=reverse)

        load_info_add_idx = getattr(self.cfg, 'load_info_add_idx', True)
        if (self.load_info is not None) and load_info_add_idx:
            for idx, info in enumerate(self.load_info):
                info['idx'] = idx

        if estimator is not None:
            self.load_info = estimator(self.load_info)

        self.try_sample = getattr(self.cfg, 'try_sample', None)
        if self.try_sample is not None:
            try: 
                start, end = self.try_sample
            except:
                start, end = 0, self.try_sample
            self.load_info = self.load_info[start:end]

        self.repeat = getattr(self.cfg, 'repeat', 1)

        pick = getattr(self.cfg, 'pick', None)
        if pick is not None:
            self.load_info = [i for i in self.load_info if i['filename'] in pick]

        #########
        # cache #
        #########

        self.cache_sm = getattr(self.cfg, 'cache_sm', False)
        self.cache_cnt = 0
        if self.cache_sm:
            self.cache_pct = getattr(self.cfg, 'cache_pct', 0)
            cache_unique_id = sync.nodewise_sync().random_sync_id()
            self.cache_unique_id = hashlib.sha256(pickle.dumps(cache_unique_id)).hexdigest()
            self.__cache__(self.cache_pct)

        #######
        # log #
        #######

        if self.load_info is not None:
            console_info = '{}: '.format(self.__class__.__name__)
            console_info += 'total {} unique images, '.format(len(self.load_info))
            console_info += 'total {} unique sample. Cached {}. Repeat {} times.'.format(
                len(self.load_info), self.cache_cnt, self.repeat)
        else:
            console_info = '{}: load_info not ready.'.format(self.__class__.__name__)
        print_log(console_info)

    def init_load_info(self):
        # implement by sub class
        pass

    def __len__(self):
        return len(self.load_info)*self.repeat

    def __cache__(self, pct):
        if pct == 0:
            self.cache_cnt = 0
            return
        self.cache_cnt = int(len(self.load_info)*pct)
        if not self.cache_sm:
            for i in range(self.cache_cnt):
                self.load_info[i] = self.loader(self.load_info[i])
            return

        for i in range(self.cache_cnt):
            shm_name = str(self.load_info[i]['unique_id']) + '_' + self.cache_unique_id
            if i % self.local_world_size == self.local_rank:
                data = pickle.dumps(self.loader(self.load_info[i]))
                datan = len(data)
                # self.print_smname_to_file(shm_name)
                shm = shared_memory.SharedMemory(
                    name=shm_name, create=True, size=datan)
                shm.buf[0:datan] = data[0:datan]
                shm.close()
                self.load_info[i] = shm_name
            else:
                self.load_info[i] = shm_name
        dist.barrier()

    def __getitem__(self, idx):
        idx = idx%len(self.load_info)
        # element = copy.deepcopy(self.load_info[idx])

        # 0730 try shared memory
        element = copy.deepcopy(self.load_info[idx])
        if isinstance(element, str):
            shm = shared_memory.SharedMemory(name=element)
            element = pickle.loads(shm.buf)
            shm.close()
        else:
            element = copy.deepcopy(element)
            element['load_info_ptr'] = self.load_info

        if idx >= self.cache_cnt:
            element = self.loader(element)
        if self.transforms is not None:
            element = self.transforms(element)
        if self.formatter is not None:
            return self.formatter(element)
        else:
            return element

    # 0730 try shared memory
    def __del__(self):
        # Clean the shared memory
        for infoi in self.load_info:
            if isinstance(infoi, str) and (self.local_rank==0):
                shm = shared_memory.SharedMemory(name=infoi)
                shm.close()
                shm.unlink()

    def print_smname_to_file(self, smname):
        try:
            log_file = cfguh().cfg.train.log_file
        except:
            try:
                log_file = cfguh().cfg.eval.log_file
            except:
                raise ValueError
        # a trick to use the log_file path
        sm_file = log_file.replace('.log', '.smname')
        with open(sm_file, 'a') as f:
            f.write(smname + '\n')

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

from .ds_loader import get_loader
from .ds_transform import get_transform
from .ds_estimator import get_estimator
from .ds_formatter import get_formatter

@singleton
class get_dataset(object):
    def __init__(self):
        self.dataset = {}

    def register(self, ds):
        self.dataset[ds.__name__] = ds

    def __call__(self, cfg):
        if cfg is None:
            return None
        t = cfg.type
        if t is None:
            return None
        elif t in ['laion2b', 'laion2b_dummy', 
                   'laion2b_webdataset', 
                   'laion2b_webdataset_sdofficial', ]:
            from .. import ds_laion2b
        elif t in ['coyo', 'coyo_dummy', 
                   'coyo_webdataset', ]:
            from .. import ds_coyo_webdataset
        elif t in ['laionart', 'laionart_dummy', 
                   'laionart_webdataset', ]:
            from .. import ds_laionart
        elif t in ['celeba']:
            from .. import ds_celeba
        elif t in ['div2k']:
            from .. import ds_div2k
        elif t in ['pafc']:
            from .. import ds_pafc
        elif t in ['coco_caption']:
            from .. import ds_coco
        else:
            raise ValueError

        loader    = get_loader()   (cfg.get('loader'   , None))
        transform = get_transform()(cfg.get('transform', None))
        estimator = get_estimator()(cfg.get('estimator', None))
        formatter = get_formatter()(cfg.get('formatter', None))

        return self.dataset[t](
            cfg, loader, estimator, 
            transform, formatter)

def register():
    def wrapper(class_):
        get_dataset().register(class_)
        return class_
    return wrapper

# some other helpers

class collate(object):
    """
        Modified from torch.utils.data._utils.collate
        It handle list different from the default.
            List collate just by append each other.
    """
    def __init__(self):
        self.default_collate = \
            torch.utils.data._utils.collate.default_collate

    def __call__(self, batch):
        """
        Args:
            batch: [data, data] -or- [(data1, data2, ...), (data1, data2, ...)]
        This function will not be used as induction function
        """
        elem = batch[0]
        if not (elem, (tuple, list)):
            return self.default_collate(batch)
        
        rv = []
        # transposed
        for i in zip(*batch):
            if isinstance(i[0], list):
                if len(i[0]) != 1:
                    raise ValueError
                try:
                    i = [[self.default_collate(ii).squeeze(0)] for ii in i]
                except:
                    pass
                rvi = list(itertools.chain.from_iterable(i))
                rv.append(rvi) # list concat
            else:
                rv.append(self.default_collate(i))
        return rv
