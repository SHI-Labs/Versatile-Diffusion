import torch
import torch.distributed as dist

import os
import os.path as osp
import numpy as np
import cv2
import copy
import json

from ..log_service import print_log

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_evaluator(object):
    def __init__(self):
        self.evaluator = {}

    def register(self, evaf, name):
        self.evaluator[name] = evaf

    def __call__(self, pipeline_cfg=None):
        if pipeline_cfg is None:
            from . import eva_null
            return self.evaluator['null']()

        if not isinstance(pipeline_cfg, list):
            t = pipeline_cfg.type
            if t == 'miou':
                from . import eva_miou
            if t == 'psnr':
                from . import eva_psnr
            if t == 'ssim':
                from . import eva_ssim
            if t == 'lpips':
                from . import eva_lpips
            if t == 'fid':
                from . import eva_fid
            return self.evaluator[t](**pipeline_cfg.args)

        evaluator = []
        for ci in pipeline_cfg:
            t = ci.type
            if t == 'miou':
                from . import eva_miou
            if t == 'psnr':
                from . import eva_psnr
            if t == 'ssim':
                from . import eva_ssim
            if t == 'lpips':
                from . import eva_lpips
            if t == 'fid':
                from . import eva_fid
            evaluator.append(
                self.evaluator[t](**ci.args))
        if len(evaluator) == 0:
            return None
        else:
            return compose(evaluator)

def register(name):
    def wrapper(class_):
        get_evaluator().register(class_, name)
        return class_
    return wrapper

class base_evaluator(object):
    def __init__(self, 
                 **args):
        '''
        Args:
            sample_n, int,
                the total number of sample. used in 
                distributed sync
        '''
        if not dist.is_available():
            raise ValueError
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.sample_n = None
        self.final = {}

    def sync(self, data):
        """
        Args:
            data: any,
                the data needs to be broadcasted
        """
        if data is None:
            return None

        if isinstance(data, tuple):
            data = list(data)

        if isinstance(data, list):
            data_list = []
            for datai in data:
                data_list.append(self.sync(datai))
            data = [[*i] for i in zip(*data_list)]
            return data

        data = [
            self.sync_(data, ranki)
                for ranki in range(self.world_size)
        ]
        return data

    def sync_(self, data, rank):

        t = type(data)
        is_broadcast = rank == self.rank

        if t is np.ndarray:
            dtrans = data
            dt = data.dtype
            if dt in [
                    int,
                    np.bool,
                    np.uint8, 
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,]:
                dtt = torch.int64
            elif dt in [
                    float,
                    np.float16,
                    np.float32,
                    np.float64,]:
                dtt = torch.float64

        elif t is str:
            dtrans = np.array(
                [ord(c) for c in data],
                dtype = np.int64
            )
            dt = np.int64
            dtt = torch.int64
        else:
            raise ValueError

        if is_broadcast:
            n = len(dtrans.shape)
            n = torch.tensor(n).long()

            n = n.to(self.rank)
            dist.broadcast(n, src=rank)

            n = list(dtrans.shape)
            n = torch.tensor(n).long()
            n = n.to(self.rank)
            dist.broadcast(n, src=rank)

            n = torch.tensor(dtrans, dtype=dtt)
            n = n.to(self.rank)
            dist.broadcast(n, src=rank)
            return data

        n = torch.tensor(0).long()
        n = n.to(self.rank)
        dist.broadcast(n, src=rank)
        n = n.item()

        n = torch.zeros(n).long()
        n = n.to(self.rank)
        dist.broadcast(n, src=rank)
        n = list(n.to('cpu').numpy())

        n = torch.zeros(n, dtype=dtt)
        n = n.to(self.rank)
        dist.broadcast(n, src=rank)
        n = n.to('cpu').numpy().astype(dt)

        if t is np.ndarray:
            return n
        elif t is str:
            n = ''.join([chr(c) for c in n])
            return n

    def zipzap_arrange(self, data):
        '''
        Order the data so it range like this:
            input [[0, 2, 4, 6], [1, 3, 5, 7]] -> output [0, 1, 2, 3, 4, 5, ...]
        '''
        if isinstance(data[0], list):
            data_new = []
            maxlen = max([len(i) for i in data])
            totlen = sum([len(i) for i in data])
            cnt = 0
            for idx in range(maxlen):
                for datai in data:
                    data_new += [datai[idx]]
                    cnt += 1
                    if cnt >= totlen:
                        break
            return data_new

        elif isinstance(data[0], np.ndarray):
            maxlen = max([i.shape[0] for i in data])
            totlen = sum([i.shape[0] for i in data])
            datai_shape = data[0].shape[1:]
            data = [
                np.concatenate(datai, np.zeros(maxlen-datai.shape[0], *datai_shape), axis=0)
                if datai.shape[0] < maxlen else datai
                for datai in data
            ] # even the array
            data = np.stack(data, axis=1).reshape(-1, *datai_shape)
            data = data[:totlen]
            return data

        else:
            raise NotImplementedError

    def add_batch(self, **args):
        raise NotImplementedError

    def set_sample_n(self, sample_n):
        self.sample_n = sample_n

    def compute(self):
        raise NotImplementedError

    # Function needed in training to judge which 
    #   evaluated number is better
    def isbetter(self, old, new):
        return new>old

    def one_line_summary(self):
        print_log('Evaluator display')

    def save(self, path):
        if not osp.exists(path):
            os.makedirs(path)
        ofile = osp.join(path, 'result.json')
        with open(ofile, 'w') as f:
            json.dump(self.final, f, indent=4)

    def clear_data(self):
        raise NotImplementedError

class compose(object):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.sample_n = None
        self.final = {}

    def add_batch(self, *args, **kwargs):
        for pi in self.pipeline:
            pi.add_batch(*args, **kwargs)

    def set_sample_n(self, sample_n):
        self.sample_n = sample_n
        for pi in self.pipeline:
            pi.set_sample_n(sample_n)

    def compute(self):
        rv = {}
        for pi in self.pipeline:
            rv[pi.symbol] = pi.compute()
            self.final[pi.symbol] = pi.final
        return rv

    def isbetter(self, old, new):
        check = 0
        for pi in self.pipeline:
            if pi.isbetter(old, new):
                check+=1
        if check/len(self.pipeline)>0.5:
            return True
        else:
            return False

    def one_line_summary(self):
        for pi in self.pipeline:
            pi.one_line_summary()

    def save(self, path):
        if not osp.exists(path):
            os.makedirs(path)
        ofile = osp.join(path, 'result.json')
        with open(ofile, 'w') as f:
            json.dump(self.final, f, indent=4)

    def clear_data(self):
        for pi in self.pipeline:
            pi.clear_data()
