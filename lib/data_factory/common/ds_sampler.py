from tokenize import group
import torch
import numpy as np
import numpy.random as npr
import torch.distributed as dist
import math

from ...log_service import print_log
from ... import sync

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_sampler(object):
    def __init__(self):
        self.sampler = {}

    def register(self, sampler):
        self.sampler[sampler.__name__] = sampler

    def __call__(self, dataset, cfg):
        if cfg == 'default_train':
            return GlobalDistributedSampler(dataset, shuffle=True, extend=False)
        elif cfg == 'default_eval':
            return GlobalDistributedSampler(dataset, shuffle=False, extend=True)
        else:
            t = cfg.type
            return self.sampler[t](dataset=dataset, **cfg.args)

def register():
    def wrapper(class_):
        get_sampler().register(class_)
        return class_
    return wrapper

######################
# DistributedSampler #
######################

@register()
class GlobalDistributedSampler(torch.utils.data.Sampler):
    """
    This is a distributed sampler that sync accross gpus and nodes.
    """
    def __init__(self, 
                 dataset, 
                 shuffle=True,
                 extend=False,):
        """
        Arguments:
            dataset: Dataset used for sampling.
            shuffle: If true, sampler will shuffle the indices
            extend: If true, sampler will extend the indices that can be even distributed by ranks 
                otherwise sampler will truncate the indices to make it even.
        """
        self.ddp = sync.is_ddp()
        self.rank = sync.get_rank('global')
        self.world_size = sync.get_world_size('global')
        self.dataset = dataset
        self.shuffle = shuffle
        self.extend = extend

        num_samples = len(dataset) // self.world_size
        if extend and (len(dataset)%self.world_size != 0):
            num_samples+=1
        self.num_samples = num_samples
        self.total_size = num_samples * self.world_size

    def __iter__(self):
        indices = self.get_sync_order()
        if self.extend:
            # extend using the front indices
            indices = indices+indices[0:self.total_size-len(indices)]
        else:
            # truncate
            indices = indices[0:self.total_size]
        # subsample
        indices = indices[self.rank : len(indices) : self.world_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def get_sync_order(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).to(self.rank)
            if self.ddp:
                dist.broadcast(indices, src=0)
            indices = indices.to('cpu').tolist()
        else:
            indices = list(range(len(self.dataset)))
        print_log('Sampler : {}'.format(str(indices[0:5])) )
        return indices

@register()
class LocalDistributedSampler(GlobalDistributedSampler):
    """
    This is a distributed sampler that sync across gpus within the nodes.
        But not sync across nodes.
    """
    def __init__(self, 
                 dataset, 
                 shuffle=True,
                 extend=False,):
        super().__init__(dataset, shuffle, extend)
        self.rank = sync.get_rank('local')
        self.world_size = sync.get_world_size('local')

    def get_sync_order(self):
        if self.shuffle:
            if self.rank == 0:
                indices = list(npr.permutation(len(self.dataset)))
                sync.nodewise_sync().broadcast_r0(indices)
            else:
                indices = sync.nodewise_sync().broadcast_r0(None)
        else:
            indices = list(range(len(self.dataset)))
        print_log('Sampler : {}'.format(str(indices[0:5])) )
        return indices

############################
# random sample with group #
############################
# Deprecated

@register()
class GroupSampler(torch.utils.data.Sampler):
    """
    This is a new DistributedSampler that sample all index according to group.
    i.e. 
    if group_size=3, num_replicas=2, train mode:
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            ==> (group) [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]
            ==> (distribute) process0: [3, 4, 5], (leftover [6, 7, 8, 9, 10])
                             process1: [0, 1, 2]
            ==> (group leftover) process0: [3, 4, 5], (leftover [6, 7], [8, 9], 10)
                                 process1: [0, 1, 2]
            ==> (distribute) process0: [3, 4, 5], [6, 7] (remove 10)
                             process1: [0, 1, 2], [8, 9]

        it will avoid_batchsize=1:
        0, 1, 2, 3, 4, 5, 6, 7, 8,
            ==> (group) [0, 1, 2], [3, 4, 5], [6, 7, 8]
            ==> (distribute) process0: [3, 4, 5], (leftover [6, 7, 8])
                             process1: [0, 1, 2]
            ==> (group leftover) process0: [3, 4, 5], (leftover [6], [7], [8])
                                 process1: [0, 1, 2]
            ==> (distribute) process0: [3, 4, 5], (remove 6, 7, 8) (because distribute make batchsize 1)
                             process1: [0, 1, 2]

    if group_size=3, num_replicas=2, eval mode:
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            ==> (extend) 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10
            ==> (group) [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 10]
            ==> (distribute) process0: [0, 1, 2], [6, 7, 8],
                             process1: [3, 4, 5], [9, 10, 10]
    """

    def __init__(self, 
                 dataset, 
                 group_size,
                 num_replicas=None, 
                 rank=None, 
                 mode='train',):
        if num_replicas is None:
            if not dist.is_available():
                raise ValueError
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise ValueError
            rank = dist.get_rank()

        self.dataset = dataset
        self.len_dataset = len(dataset)
        self.group_size = group_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.mode = mode
        len_dataset = self.len_dataset

        if (len_dataset % num_replicas != 0) and (mode == 'train'):
            # drop the non_aligned
            aligned_indices = np.arange(len_dataset)[:-(len_dataset % num_replicas)]
            aligned_len_dataset = aligned_indices.shape[0]
        elif (len_dataset % num_replicas != 0) and (mode == 'eval'):
            extend = np.array([len_dataset-1 for _ in range(num_replicas - len_dataset % num_replicas)])
            aligned_indices = np.concatenate([range(len_dataset), extend])
            aligned_len_dataset = aligned_indices.shape[0]
        else:
            aligned_indices = np.arange(len_dataset)
            aligned_len_dataset = len_dataset

        num_even_distributed_groups = aligned_len_dataset // (group_size * num_replicas)
        num_even = num_even_distributed_groups * group_size * num_replicas

        self.regular_groups = aligned_indices[0:num_even].reshape(-1, group_size)
        self.leftover_groups = aligned_indices[num_even:].reshape(num_replicas, -1)

        if self.leftover_groups.size == 0:
            self.leftover_groups = None
        elif (self.leftover_groups.shape[-1]==1) and (mode == 'train'):
            # avoid bs=1
            self.leftover_groups = None

        # a urly way to modify dataset.load_info according to the grouping
        for groupi in self.regular_groups:
            for idx in groupi:
                idx_lowerbd = groupi[0]
                idx_upperbd = groupi[-1]
                idx_reference = (idx_lowerbd+idx_upperbd)//2
                dataset.load_info[idx]['ref_size'] = dataset.load_info[idx_reference]['image_size']
        if self.leftover_groups is not None:
            for groupi in self.leftover_groups:
                for idx in groupi:
                    idx_lowerbd = groupi[0]
                    idx_upperbd = groupi[-1]
                    idx_reference = (idx_lowerbd+idx_upperbd)//2
                    dataset.load_info[idx]['ref_size'] = dataset.load_info[idx_reference]['image_size']

    def concat(self, nparrays, axis=0):
        # a helper for save concaternation
        nparrays = [i for i in nparrays if i.size > 0]
        return np.concatenate(nparrays, axis=axis)

    def __iter__(self):
        indices = self.get_sync_order()
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def get_sync_order(self):
        # g = torch.Generator()
        # g.manual_seed(self.epoch)

        mode         = self.mode
        rank         = self.rank
        num_replicas = self.num_replicas
        group_size   = self.group_size
        num_groups = len(self.regular_groups)

        if mode == 'train':
            g_indices = torch.randperm(num_groups).to(rank)
            dist.broadcast(g_indices, src=0)
            g_indices = g_indices.to('cpu').tolist()
            num_groups_per_rank = num_groups // num_replicas
            groups = self.regular_groups[g_indices][num_groups_per_rank*rank : num_groups_per_rank*(rank+1)]
            indices = groups.flatten()

            if self.leftover_groups is not None:
                leftg_indices = torch.randperm(len(self.leftover_groups)).to(rank)
                dist.broadcast(leftg_indices, src=0)
                leftg_indices = leftg_indices.to('cpu').tolist()
                last = self.leftover_groups[leftg_indices][rank]
                indices = np.concatenate([indices, last], axis=0)
        elif mode == 'eval':
            groups = self.regular_groups.reshape(-1, num_replicas, group_size)[:, rank, :]
            indices = groups.flatten()
            if self.leftover_groups is not None:
                last = self.leftover_groups[rank]
                indices = np.concatenate([indices, last], axis=0)
        else:
            raise ValueError
        
        print_log('Sampler RANK {} : {}'.format(rank, str(indices[0:group_size+1])))
        return indices
