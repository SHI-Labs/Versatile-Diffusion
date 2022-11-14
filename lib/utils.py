import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# cudnn.enabled = True
# cudnn.benchmark = True
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import os.path as osp
import sys
import numpy as np
import pprint
import timeit
import time
import copy
import matplotlib.pyplot as plt

from .cfg_holder import cfg_unique_holder as cfguh

from .data_factory import \
    get_dataset, collate, \
    get_loader, \
    get_transform, \
    get_estimator, \
    get_formatter, \
    get_sampler

from .model_zoo import \
    get_model, get_optimizer, get_scheduler

from .log_service import print_log, distributed_log_manager

from .evaluator import get_evaluator
from . import sync

class train_stage(object):
    """
    This is a template for a train stage,
        (can be either train or test or anything)
    Usually, it takes RANK
        one dataloader, one model, one optimizer, one scheduler.
    But it is not limited to these parameters. 
    """
    def __init__(self):
        self.nested_eval_stage = None
        self.rv_keep = None

    def is_better(self, x):
        return (self.rv_keep is None) or (x>self.rv_keep)

    def set_model(self, net, mode):
        if mode == 'train':
            return net.train()
        elif mode == 'eval':
            return net.eval()
        else:
            raise ValueError

    def __call__(self,
                 **paras):
        cfg = cfguh().cfg
        cfgt = cfg.train
        logm = distributed_log_manager()
        epochn, itern, samplen = 0, 0, 0

        step_type = cfgt.get('step_type', 'iter')
        assert step_type in ['epoch', 'iter', 'sample'], \
            'Step type must be in [epoch, iter, sample]'
 
        step_num      = cfgt.get('step_num'     , None)
        gradacc_every = cfgt.get('gradacc_every', 1   )
        log_every     = cfgt.get('log_every'    , None)
        ckpt_every    = cfgt.get('ckpt_every'   , None)
        eval_start    = cfgt.get('eval_start'   , 0   )
        eval_every    = cfgt.get('eval_every'   , None)

        if paras.get('resume_step', None) is not None:
            resume_step = paras['resume_step']
            assert step_type == resume_step['type']
            epochn = resume_step['epochn']
            itern = resume_step['itern']
            samplen = resume_step['samplen']
            del paras['resume_step']

        trainloader = paras['trainloader']
        optimizer   = paras['optimizer']
        scheduler   = paras['scheduler']
        net         = paras['net']

        GRANK, LRANK, NRANK = sync.get_rank('all')
        GWSIZE, LWSIZE, NODES = sync.get_world_size('all')

        weight_path = osp.join(cfgt.log_dir, 'weight')
        if (GRANK==0) and (not osp.isdir(weight_path)):
            os.makedirs(weight_path)
        if (GRANK==0) and (cfgt.save_init_model):
            self.save(net, is_init=True, step=0, optimizer=optimizer)

        epoch_time = timeit.default_timer()
        end_flag = False 
        net.train()

        while True:
            if step_type == 'epoch':
                lr = scheduler[epochn] if scheduler is not None else None
            for batch in trainloader:
                # so first element of batch (usually image) can be [tensor]
                if not isinstance(batch[0], list):
                    bs = batch[0].shape[0]
                else:
                    bs = len(batch[0])
                if cfgt.skip_partial_batch and (bs != cfgt.batch_size_per_gpu):
                    continue

                itern_next = itern + 1
                samplen_next = samplen + bs*GWSIZE

                if step_type == 'iter':
                    lr = scheduler[itern//gradacc_every] if scheduler is not None else None
                    grad_update = itern%gradacc_every==(gradacc_every-1)
                elif step_type == 'sample':
                    lr = scheduler[samplen] if scheduler is not None else None
                    # TODO: 
                    # grad_update = samplen%gradacc_every==(gradacc_every-1) 

                # timeDebug = timeit.default_timer()
                paras_new = self.main(
                    batch=batch, 
                    lr=lr,
                    itern=itern,
                    epochn=epochn,
                    samplen=samplen,
                    isinit=False,
                    grad_update=grad_update,
                    **paras)
                # print_log(timeit.default_timer() - timeDebug)

                paras.update(paras_new)
                logm.accumulate(bs, **paras['log_info'])

                #######
                # log #
                #######

                display_flag = False
                if log_every is not None:
                    display_i = (itern//log_every) != (itern_next//log_every)
                    display_s = (samplen//log_every) != (samplen_next//log_every)
                    display_flag = (display_i and (step_type=='iter')) \
                        or (display_s and (step_type=='sample'))

                if display_flag:
                    tbstep = itern_next if step_type=='iter' else samplen_next
                    console_info = logm.train_summary(
                        itern_next, epochn, samplen_next, lr, tbstep=tbstep)
                    logm.clear()
                    print_log(console_info)

                ########
                # eval #
                ########

                eval_flag = False
                if (self.nested_eval_stage is not None) and (eval_every is not None) and (NRANK == 0):
                    if step_type=='iter':
                        eval_flag = (itern//eval_every) != (itern_next//eval_every)
                        eval_flag = eval_flag and (itern_next>=eval_start)
                        eval_flag = eval_flag or itern==0
                    if step_type=='sample':
                        eval_flag = (samplen//eval_every) != (samplen_next//eval_every)
                        eval_flag = eval_flag and (samplen_next>=eval_start)
                        eval_flag = eval_flag or samplen==0

                if eval_flag:
                    eval_cnt = itern_next if step_type=='iter' else samplen_next
                    net = self.set_model(net, 'eval')
                    rv = self.nested_eval_stage(
                        eval_cnt=eval_cnt, **paras)
                    rv = rv.get('eval_rv', None)
                    if rv is not None:
                        logm.tensorboard_log(eval_cnt, rv, mode='eval')
                    if self.is_better(rv):
                        self.rv_keep = rv
                        if GRANK==0:
                            step = {'epochn':epochn, 'itern':itern_next, 
                                    'samplen':samplen_next, 'type':step_type, }
                            self.save(net, is_best=True, step=step, optimizer=optimizer)
                    net = self.set_model(net, 'train')

                ########
                # ckpt # 
                ########

                ckpt_flag = False
                if (GRANK==0) and (ckpt_every is not None):
                    # not distributed
                    ckpt_i = (itern//ckpt_every) != (itern_next//ckpt_every)
                    ckpt_s = (samplen//ckpt_every) != (samplen_next//ckpt_every)
                    ckpt_flag = (ckpt_i and (step_type=='iter')) \
                        or (ckpt_s and (step_type=='sample'))

                if ckpt_flag:
                    if step_type == 'iter':
                        print_log('Checkpoint... {}'.format(itern_next))
                        step = {'epochn':epochn, 'itern':itern_next, 
                                'samplen':samplen_next, 'type':step_type, }
                        self.save(net, itern=itern_next, step=step, optimizer=optimizer)
                    else:
                        print_log('Checkpoint... {}'.format(samplen_next))
                        step = {'epochn':epochn, 'itern':itern_next, 
                                'samplen':samplen_next, 'type':step_type, }
                        self.save(net, samplen=samplen_next, step=step, optimizer=optimizer)

                #######
                # end #
                #######

                itern = itern_next
                samplen = samplen_next

                if step_type is not None:
                    end_flag = (itern>=step_num and (step_type=='iter')) \
                        or (samplen>=step_num and (step_type=='sample'))
                if end_flag:
                    break
                # loop end

            epochn += 1
            print_log('Epoch {} time:{:.2f}s.'.format(
                epochn, timeit.default_timer()-epoch_time))
            epoch_time = timeit.default_timer()

            if end_flag:
                break
            elif step_type != 'epoch':
                # This is temporarily added to resolve the data issue
                trainloader = self.trick_update_trainloader(trainloader)
                continue

            #######
            # log #
            #######

            display_flag = False
            if (log_every is not None) and (step_type=='epoch'):
                display_flag = (epochn==1) or (epochn%log_every==0)

            if display_flag:
                console_info = logm.train_summary(
                    itern, epochn, samplen, lr, tbstep=epochn)
                logm.clear()
                print_log(console_info)

            ########
            # eval #
            ########

            eval_flag = False
            if (self.nested_eval_stage is not None) and (eval_every is not None) \
                    and (step_type=='epoch') and (NRANK==0):
                eval_flag = (epochn%eval_every==0) and (itern_next>=eval_start)
                eval_flag = (epochn==1) or eval_flag

            if eval_flag:
                net = self.set_model(net, 'eval')
                rv = self.nested_eval_stage(
                    eval_cnt=epochn,
                    **paras)['eval_rv']
                if rv is not None:
                    logm.tensorboard_log(epochn, rv, mode='eval')
                if self.is_better(rv):
                    self.rv_keep = rv
                    if (GRANK==0):
                        step = {'epochn':epochn, 'itern':itern, 
                                'samplen':samplen, 'type':step_type, }
                        self.save(net, is_best=True, step=step, optimizer=optimizer)
                net = self.set_model(net, 'train')

            ########
            # ckpt # 
            ########

            ckpt_flag = False
            if (ckpt_every is not None) and (GRANK==0) and (step_type=='epoch'):
                # not distributed
                ckpt_flag = epochn%ckpt_every==0

            if ckpt_flag:
                print_log('Checkpoint... {}'.format(itern_next))
                step = {'epochn':epochn, 'itern':itern, 
                        'samplen':samplen, 'type':step_type, }
                self.save(net, epochn=epochn, step=step, optimizer=optimizer)

            #######
            # end #
            #######
            if (step_type=='epoch') and (epochn>=step_num):
                break
            # loop end

            # This is temporarily added to resolve the data issue
            trainloader = self.trick_update_trainloader(trainloader)

        logm.tensorboard_close()
        return {}

    def main(self, **paras):
        raise NotImplementedError

    def trick_update_trainloader(self, trainloader):
        return trainloader

    def save_model(self, net, path_noext, **paras):
        cfgt = cfguh().cfg.train
        path = path_noext+'.pth'
        if isinstance(net, (torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel)):
            netm = net.module
        else:
            netm = net
        torch.save(netm.state_dict(), path)
        print_log('Saving model file {0}'.format(path))

    def save(self, net, itern=None, epochn=None, samplen=None, 
             is_init=False, is_best=False, is_last=False, **paras):
        exid = cfguh().cfg.env.experiment_id
        cfgt = cfguh().cfg.train
        cfgm = cfguh().cfg.model
        if isinstance(net, (torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel)):
            netm = net.module
        else:
            netm = net
        net_symbol = cfgm.symbol

        check = sum([
            itern is not None, samplen is not None, epochn is not None, 
            is_init, is_best, is_last])
        assert check<2

        if itern is not None:
            path_noexp = '{}_{}_iter_{}'.format(exid, net_symbol, itern)
        elif samplen is not None:
            path_noexp = '{}_{}_samplen_{}'.format(exid, net_symbol, samplen)
        elif epochn is not None:
            path_noexp = '{}_{}_epoch_{}'.format(exid, net_symbol, epochn)
        elif is_init:
            path_noexp = '{}_{}_init'.format(exid, net_symbol)
        elif is_best:
            path_noexp = '{}_{}_best'.format(exid, net_symbol)
        elif is_last:
            path_noexp = '{}_{}_last'.format(exid, net_symbol)
        else:
            path_noexp = '{}_{}_default'.format(exid, net_symbol)

        path_noexp = osp.join(cfgt.log_dir, 'weight', path_noexp)
        self.save_model(net, path_noexp, **paras)

class eval_stage(object):
    def __init__(self):
        self.evaluator = None

    def create_dir(self, path):
        local_rank = sync.get_rank('local')
        if (not osp.isdir(path)) and (local_rank == 0):
            os.makedirs(path)
        sync.nodewise_sync().barrier()

    def __call__(self, 
                 evalloader,
                 net,
                 **paras):
        cfgt = cfguh().cfg.eval
        local_rank = sync.get_rank('local')
        if self.evaluator is None:
            evaluator = get_evaluator()(cfgt.evaluator)
            self.evaluator = evaluator
        else:
            evaluator = self.evaluator

        time_check = timeit.default_timer()

        for idx, batch in enumerate(evalloader): 
            rv = self.main(batch, net)
            evaluator.add_batch(**rv)
            if cfgt.output_result:
                try:
                    self.output_f(**rv, cnt=paras['eval_cnt'])
                except:
                    self.output_f(**rv)
            if idx%cfgt.log_display == cfgt.log_display-1:
                print_log('processed.. {}, Time:{:.2f}s'.format(
                    idx+1, timeit.default_timer() - time_check))
                time_check = timeit.default_timer()
            # break

        evaluator.set_sample_n(len(evalloader.dataset))
        eval_rv = evaluator.compute()
        if local_rank == 0:
            evaluator.one_line_summary()
            evaluator.save(cfgt.log_dir)
        evaluator.clear_data()
        return {
            'eval_rv' : eval_rv
        }

class exec_container(object):
    """
    This is the base functor for all types of executions.
        One execution can have multiple stages, 
        but are only allowed to use the same 
        config, network, dataloader. 
    Thus, in most of the cases, one exec_container is one
        training/evaluation/demo...
    If DPP is in use, this functor should be spawn.
    """
    def __init__(self,
                 cfg,
                 **kwargs):
        self.cfg = cfg
        self.registered_stages = []
        self.node_rank = None
        self.local_rank = None
        self.global_rank = None
        self.local_world_size = None
        self.global_world_size = None
        self.nodewise_sync_global_obj = sync.nodewise_sync_global()

    def register_stage(self, stage):
        self.registered_stages.append(stage)

    def __call__(self, 
                 local_rank,
                 **kwargs):
        cfg = self.cfg
        cfguh().save_cfg(cfg)

        self.node_rank = cfg.env.node_rank
        self.local_rank = local_rank
        self.nodes = cfg.env.nodes
        self.local_world_size = cfg.env.gpu_count

        self.global_rank = self.local_rank + self.node_rank * self.nodes
        self.global_world_size = self.nodes * self.local_world_size

        dist.init_process_group(
            backend = cfg.env.dist_backend,
            init_method = cfg.env.dist_url,
            rank = self.global_rank,
            world_size = self.global_world_size,)
        torch.cuda.set_device(local_rank)
        sync.nodewise_sync().copy_global(self.nodewise_sync_global_obj).local_init()
        
        if isinstance(cfg.env.rnd_seed, int):
            np.random.seed(cfg.env.rnd_seed + self.global_rank)
            torch.manual_seed(cfg.env.rnd_seed + self.global_rank)

        time_start = timeit.default_timer()

        para = {'itern_total' : 0,}
        dl_para = self.prepare_dataloader()
        assert isinstance(dl_para, dict)
        para.update(dl_para)

        md_para = self.prepare_model()
        assert isinstance(md_para, dict)
        para.update(md_para)

        for stage in self.registered_stages:
            stage_para = stage(**para)
            if stage_para is not None:
                para.update(stage_para)

        if self.global_rank==0:
            self.save_last_model(**para)

        print_log(
            'Total {:.2f} seconds'.format(timeit.default_timer() - time_start))
        dist.destroy_process_group()

    def prepare_dataloader(self):
        """
        Prepare the dataloader from config.
        """
        return {
            'trainloader' : None,
            'evalloader' : None}

    def prepare_model(self):
        """
        Prepare the model from config.
        """
        return {'net' : None}

    def save_last_model(self, **para):
        return

    def destroy(self):
        self.nodewise_sync_global_obj.destroy()

class train(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        trainset = get_dataset()(cfg.train.dataset)
        sampler = get_sampler()(
            dataset=trainset, cfg=cfg.train.dataset.get('sampler', 'default_train'))
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size = cfg.train.batch_size_per_gpu, 
            sampler = sampler, 
            num_workers = cfg.train.dataset_num_workers_per_gpu, 
            drop_last = False, 
            pin_memory = cfg.train.dataset.get('pin_memory', False),
            collate_fn = collate(),)

        evalloader = None
        if 'eval' in cfg:
            evalset = get_dataset()(cfg.eval.dataset)
            if evalset is not None:
                sampler = get_sampler()(
                    dataset=evalset, cfg=cfg.eval.dataset.get('sampler', 'default_eval'))
                evalloader = torch.utils.data.DataLoader(
                    evalset, 
                    batch_size = cfg.eval.batch_size_per_gpu,
                    sampler = sampler,
                    num_workers = cfg.eval.dataset_num_workers_per_gpu,
                    drop_last = False, 
                    pin_memory = cfg.eval.dataset.get('pin_memory', False),
                    collate_fn = collate(),)
            
        return {
            'trainloader' : trainloader,
            'evalloader'  : evalloader,}

    def prepare_model(self):
        cfg = cfguh().cfg
        net = get_model()(cfg.model)
        if cfg.env.cuda:
            net.to(self.local_rank)
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[self.local_rank], 
                find_unused_parameters=True)
        net.train() 
        scheduler = get_scheduler()(cfg.train.scheduler)
        optimizer = get_optimizer()(net, cfg.train.optimizer)
        return {
            'net'       : net,
            'optimizer' : optimizer,
            'scheduler' : scheduler,}

    def save_last_model(self, **para):
        cfgt = cfguh().cfg.train
        net = para['net']
        net_symbol = cfguh().cfg.model.symbol
        if isinstance(net, (torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel)):
            netm = net.module
        else:
            netm = net
        path = osp.join(cfgt.log_dir, '{}_{}_last.pth'.format(
            cfgt.experiment_id, net_symbol))
        torch.save(netm.state_dict(), path)
        print_log('Saving model file {0}'.format(path))

class eval(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        evalloader = None
        if cfg.eval.get('dataset', None) is not None:
            evalset = get_dataset()(cfg.eval.dataset)
            if evalset is None:
                return
            sampler = get_sampler()(
                dataset=evalset, cfg=getattr(cfg.eval.dataset, 'sampler', 'default_eval'))
            evalloader = torch.utils.data.DataLoader(
                evalset, 
                batch_size = cfg.eval.batch_size_per_gpu,
                sampler = sampler,
                num_workers = cfg.eval.dataset_num_workers_per_gpu,
                drop_last = False, 
                pin_memory = False,
                collate_fn = collate(), )
        return {
            'trainloader' : None,
            'evalloader'  : evalloader,}

    def prepare_model(self):
        cfg = cfguh().cfg
        net = get_model()(cfg.model)
        if cfg.env.cuda:
            net.to(self.local_rank)
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[self.local_rank], 
                find_unused_parameters=True)
        net.eval()
        return {'net' : net,}

    def save_last_model(self, **para):
        return

###############
# some helper #
###############

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

import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
