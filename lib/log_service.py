import timeit
import numpy as np
import os
import os.path as osp
import shutil
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from .cfg_holder import cfg_unique_holder as cfguh
from . import sync

print_console_local_rank0_only = True

def print_log(*console_info):
    local_rank = sync.get_rank('local')
    if print_console_local_rank0_only and (local_rank!=0):
        return
    console_info = [str(i) for i in console_info]
    console_info = ' '.join(console_info)
    print(console_info)

    if local_rank!=0:
        return

    log_file = None
    try:
        log_file = cfguh().cfg.train.log_file
    except:
        try:
            log_file = cfguh().cfg.eval.log_file
        except:
            return
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(console_info + '\n')

class distributed_log_manager(object):
    def __init__(self):
        self.sum = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()

        cfgt = cfguh().cfg.train
        use_tensorboard = getattr(cfgt, 'log_tensorboard', False)

        self.ddp = sync.is_ddp()
        self.rank = sync.get_rank('local')
        self.world_size = sync.get_world_size('local')

        self.tb = None
        if use_tensorboard and (self.rank==0):
            import tensorboardX
            monitoring_dir = osp.join(cfguh().cfg.train.log_dir, 'tensorboard')
            self.tb = tensorboardX.SummaryWriter(osp.join(monitoring_dir))

    def accumulate(self, n, **data):
        if n < 0:
            raise ValueError

        for itemn, di in data.items():
            if itemn in self.sum:
                self.sum[itemn] += di * n
                self.cnt[itemn] += n
            else:
                self.sum[itemn] = di * n
                self.cnt[itemn] = n

    def get_mean_value_dict(self):
        value_gather = [
            self.sum[itemn]/self.cnt[itemn] \
                for itemn in sorted(self.sum.keys()) ]

        value_gather_tensor = torch.FloatTensor(value_gather).to(self.rank)
        if self.ddp:
            dist.all_reduce(value_gather_tensor, op=dist.ReduceOp.SUM)
            value_gather_tensor /= self.world_size

        mean = {}
        for idx, itemn in enumerate(sorted(self.sum.keys())):
            mean[itemn] = value_gather_tensor[idx].item()
        return mean

    def tensorboard_log(self, step, data, mode='train', **extra):
        if self.tb is None:
            return
        if mode == 'train':
            self.tb.add_scalar('other/epochn', extra['epochn'], step)
            if 'lr' in extra:
                self.tb.add_scalar('other/lr', extra['lr'], step)
            for itemn, di in data.items():
                if itemn.find('loss') == 0:
                    self.tb.add_scalar('loss/'+itemn,  di, step)
                elif itemn == 'Loss':
                    self.tb.add_scalar('Loss',  di, step)
                else:
                    self.tb.add_scalar('other/'+itemn, di, step)
        elif mode == 'eval':
            if isinstance(data, dict):
                for itemn, di in data.items():
                    self.tb.add_scalar('eval/'+itemn, di, step)
            else:
                self.tb.add_scalar('eval', data, step)
        return

    def train_summary(self, itern, epochn, samplen, lr, tbstep=None):
        console_info = [
            'Iter:{}'.format(itern),
            'Epoch:{}'.format(epochn),
            'Sample:{}'.format(samplen),]

        if lr is not None:
            console_info += ['LR:{:.4E}'.format(lr)]

        mean = self.get_mean_value_dict()

        tbstep = itern if tbstep is None else tbstep
        self.tensorboard_log(
            tbstep, mean, mode='train',
            itern=itern, epochn=epochn, lr=lr)

        loss = mean.pop('Loss')
        mean_info = ['Loss:{:.4f}'.format(loss)] + [
            '{}:{:.4f}'.format(itemn, mean[itemn]) \
                for itemn in sorted(mean.keys()) \
                    if itemn.find('loss') == 0
        ]
        console_info += mean_info
        console_info.append('Time:{:.2f}s'.format(
            timeit.default_timer() - self.time_check))
        return ' , '.join(console_info)

    def clear(self):
        self.sum = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()

    def tensorboard_close(self):
        if self.tb is not None:
            self.tb.close()

# ----- also include some small utils -----

def torch_to_numpy(*argv):
    if len(argv) > 1:
        data = list(argv)
    else:
        data = argv[0]

    if isinstance(data, torch.Tensor):
        return data.to('cpu').detach().numpy()

    elif isinstance(data, (list, tuple)):
        out = []
        for di in data:
            out.append(torch_to_numpy(di))
        return out

    elif isinstance(data, dict):
        out = {}
        for ni, di in data.items():
            out[ni] = torch_to_numpy(di)
        return out
    
    else:
        return data
