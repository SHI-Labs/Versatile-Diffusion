import torch
import torch.distributed as dist
from torchvision import transforms as tvtrans
import os
import os.path as osp
import time
import timeit
import copy
import json
import pickle
import PIL.Image
import numpy as np
from datetime import datetime
from easydict import EasyDict as edict
from collections import OrderedDict

from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.data_factory import get_dataset, get_sampler, collate
from lib.model_zoo import \
    get_model, get_optimizer, get_scheduler
from lib.log_service import print_log

from ..utils import train as train_base
from ..utils import eval as eval_base
from ..utils import train_stage as tsbase
from ..utils import eval_stage as esbase
from .. import sync

###############
# some helper #
###############

def atomic_save(cfg, net, opt, step, path):
    if isinstance(net, (torch.nn.DataParallel,
                        torch.nn.parallel.DistributedDataParallel)):
        netm = net.module
    else:
        netm = net
    sd = netm.state_dict()
    slimmed_sd = [(ki, vi) for ki, vi in sd.items()
        if ki.find('first_stage_model')!=0 and ki.find('cond_stage_model')!=0]

    checkpoint = {
        "config" : cfg,
        "state_dict" : OrderedDict(slimmed_sd),
        "step" : step}
    if opt is not None:
        checkpoint['optimizer_states'] = opt.state_dict()
    import io
    import fsspec
    bytesbuffer = io.BytesIO()
    torch.save(checkpoint, bytesbuffer)
    with fsspec.open(path, "wb") as f:
        f.write(bytesbuffer.getvalue())

def load_state_dict(net, cfg):
    pretrained_pth_full  = cfg.get('pretrained_pth_full' , None)
    pretrained_ckpt_full = cfg.get('pretrained_ckpt_full', None)
    pretrained_pth       = cfg.get('pretrained_pth'      , None)
    pretrained_ckpt      = cfg.get('pretrained_ckpt'     , None)
    pretrained_pth_dm    = cfg.get('pretrained_pth_dm'   , None)
    pretrained_pth_ema   = cfg.get('pretrained_pth_ema'  , None)
    strict_sd = cfg.get('strict_sd', False)
    errmsg = "Overlapped model state_dict! This is undesired behavior!"

    if pretrained_pth_full is not None or pretrained_ckpt_full is not None:
        assert (pretrained_pth is None) and \
               (pretrained_ckpt is None) and \
               (pretrained_pth_dm is None) and \
               (pretrained_pth_ema is None), errmsg            
        if pretrained_pth_full is not None:
            target_file = pretrained_pth_full
            sd = torch.load(target_file, map_location='cpu')
            assert pretrained_ckpt is None, errmsg
        else:
            target_file = pretrained_ckpt_full
            sd = torch.load(target_file, map_location='cpu')['state_dict']
        print_log('Load full model from [{}] strict [{}].'.format(
            target_file, strict_sd))
        net.load_state_dict(sd, strict=strict_sd)

    if pretrained_pth is not None or pretrained_ckpt is not None:
        assert (pretrained_ckpt_full is None) and \
               (pretrained_pth_full is None) and \
               (pretrained_pth_dm is None) and \
               (pretrained_pth_ema is None), errmsg
        if pretrained_pth is not None:
            target_file = pretrained_pth
            sd = torch.load(target_file, map_location='cpu')
            assert pretrained_ckpt is None, errmsg
        else:
            target_file = pretrained_ckpt
            sd = torch.load(target_file, map_location='cpu')['state_dict']
        print_log('Load model from [{}] strict [{}].'.format(
            target_file, strict_sd))
        sd_extra = [(ki, vi) for ki, vi in net.state_dict().items() \
            if ki.find('first_stage_model')==0 or ki.find('cond_stage_model')==0]
        sd.update(OrderedDict(sd_extra))
        net.load_state_dict(sd, strict=strict_sd)

    if pretrained_pth_dm is not None:
        assert (pretrained_ckpt_full is None) and \
               (pretrained_pth_full is None) and \
               (pretrained_pth is None) and \
               (pretrained_ckpt is None), errmsg
        print_log('Load diffusion model from [{}] strict [{}].'.format(
            pretrained_pth_dm, strict_sd))
        sd = torch.load(pretrained_pth_dm, map_location='cpu')
        net.model.diffusion_model.load_state_dict(sd, strict=strict_sd)

    if pretrained_pth_ema is not None:
        assert (pretrained_ckpt_full is None) and \
               (pretrained_pth_full is None) and \
               (pretrained_pth is None) and \
               (pretrained_ckpt is None), errmsg
        print_log('Load unet ema model from [{}] strict [{}].'.format(
            pretrained_pth_ema, strict_sd))
        sd = torch.load(pretrained_pth_ema, map_location='cpu')
        net.model_ema.load_state_dict(sd, strict=strict_sd)

def auto_merge_imlist(imlist, max=64):
    imlist = imlist[0:max]
    h, w = imlist[0].shape[0:2]
    num_images = len(imlist)
    num_row = int(np.sqrt(num_images))
    num_col = num_images//num_row + 1 if num_images%num_row!=0 else num_images//num_row
    canvas = np.zeros([num_row*h, num_col*w, 3], dtype=np.uint8)
    for idx, im in enumerate(imlist):
        hi = (idx // num_col) * h
        wi = (idx % num_col) * w
        canvas[hi:hi+h, wi:wi+w, :] = im
    return canvas

def latent2im(net, latent):
    single_input = len(latent.shape) == 3
    if single_input:
        latent = latent[None]
    im = net.decode_image(latent.to(net.device))
    im = torch.clamp((im+1.0)/2.0, min=0.0, max=1.0)
    im = [tvtrans.ToPILImage()(i) for i in im]
    if single_input:
        im = im[0]
    return im

def im2latent(net, im):
    single_input = not isinstance(im, list)
    if single_input:
        im = [im]
    im = torch.stack([tvtrans.ToTensor()(i) for i in im], dim=0)
    im = (im*2-1).to(net.device)
    z = net.encode_image(im)
    if single_input:
        z = z[0]
    return z

class color_adjust(object):
    def __init__(self, ref_from, ref_to):
        x0, m0, std0 = self.get_data_and_stat(ref_from)
        x1, m1, std1 = self.get_data_and_stat(ref_to)
        self.ref_from_stat = (m0, std0)
        self.ref_to_stat   = (m1, std1)
        self.ref_from = self.preprocess(x0).reshape(-1, 3)
        self.ref_to = x1.reshape(-1, 3)

    def get_data_and_stat(self, x):
        if isinstance(x, str):
            x = np.array(PIL.Image.open(x))
        elif isinstance(x, PIL.Image.Image):
            x = np.array(x)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, min=0.0, max=1.0)
            x = np.array(tvtrans.ToPILImage()(x))
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError
        x = x.astype(float)
        m = np.reshape(x, (-1, 3)).mean(0)
        s = np.reshape(x, (-1, 3)).std(0)
        return x, m, s

    def preprocess(self, x):
        m0, s0 = self.ref_from_stat
        m1, s1 = self.ref_to_stat
        y = ((x-m0)/s0)*s1 + m1
        return y

    def __call__(self, xin, keep=0, simple=False):
        xin, _, _ = self.get_data_and_stat(xin)
        x = self.preprocess(xin)
        if simple: 
            y = (x*(1-keep) + xin*keep)
            y = np.clip(y, 0, 255).astype(np.uint8)
            return y

        h, w = x.shape[:2]
        x = x.reshape(-1, 3)
        y = []
        for chi in range(3):
            yi = self.pdf_transfer_1d(self.ref_from[:, chi], self.ref_to[:, chi], x[:, chi])
            y.append(yi)

        y = np.stack(y, axis=1)
        y = y.reshape(h, w, 3)
        y = (y.astype(float)*(1-keep) + xin.astype(float)*keep)
        y = np.clip(y, 0, 255).astype(np.uint8)
        return y

    def pdf_transfer_1d(self, arr_fo, arr_to, arr_in, n=600):
        arr = np.concatenate((arr_fo, arr_to))
        min_v = arr.min() - 1e-6
        max_v = arr.max() + 1e-6
        min_vto = arr_to.min() - 1e-6
        max_vto = arr_to.max() + 1e-6
        xs = np.array(
            [min_v + (max_v - min_v) * i / n for i in range(n + 1)])
        hist_fo, _ = np.histogram(arr_fo, xs)
        hist_to, _ = np.histogram(arr_to, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_fo = np.cumsum(hist_fo)
        cum_to = np.cumsum(hist_to)
        d_fo = cum_fo / cum_fo[-1]
        d_to = cum_to / cum_to[-1]
        # transfer
        t_d = np.interp(d_fo, d_to, xs)
        t_d[d_fo <= d_to[ 0]] = min_vto
        t_d[d_fo >= d_to[-1]] = max_vto
        arr_out = np.interp(arr_in, xs, t_d)
        return arr_out

########
# main #
########

class eval(eval_base):
    def prepare_model(self):
        cfg = cfguh().cfg
        net = get_model()(cfg.model)
        if cfg.env.cuda:
            net.to(self.local_rank)
            load_state_dict(net, cfg.eval) #<--- added
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[self.local_rank], 
                find_unused_parameters=True)
        net.eval()
        return {'net' : net,}

class eval_stage(esbase):
    """
    This is eval stage that can check comprehensive results
    """
    def __init__(self):
        from ..model_zoo.ddim import DDIMSampler
        self.sampler = DDIMSampler

    def get_net(self, paras):
        return paras['net']

    def get_image_path(self):
        if 'train' in cfguh().cfg:
            log_dir = cfguh().cfg.train.log_dir
        else:
            log_dir = cfguh().cfg.eval.log_dir
        return os.path.join(log_dir, "udemo")

    @torch.no_grad()
    def sample(self, net, sampler, prompt, output_dim, scale, n_samples, ddim_steps, ddim_eta):
        h, w = output_dim
        uc = None
        if scale != 1.0:
            uc = net.get_learned_conditioning(n_samples * [""])
        c = net.get_learned_conditioning(n_samples * [prompt])
        shape = [4, h//8, w//8]
        rv = sampler.sample(
            S=ddim_steps,
            conditioning=c,
            batch_size=n_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta)
        return rv

    def save_images(self, pil_list, name, path, suffix=''):
        canvas = auto_merge_imlist([np.array(i) for i in pil_list])
        image_name = '{}{}.png'.format(name, suffix)
        PIL.Image.fromarray(canvas).save(osp.join(path, image_name))

    def __call__(self, **paras):
        cfg = cfguh().cfg
        cfgv = cfg.eval

        net = paras['net']
        eval_cnt = paras.get('eval_cnt', None)
        fix_seed = cfgv.get('fix_seed', False)

        LRANK = sync.get_rank('local')
        LWSIZE = sync.get_world_size('local')

        image_path = self.get_image_path()
        self.create_dir(image_path)
        eval_cnt = paras.get('eval_cnt', None)
        suffix='' if eval_cnt is None else '_itern'+str(eval_cnt)

        if isinstance(net, (torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel)):
            netm = net.module
        else:
            netm = net

        with_ema = getattr(netm, 'model_ema', None) is not None
        sampler = self.sampler(netm)
        setattr(netm, 'device', LRANK) # Trick

        replicate = cfgv.get('replicate', 1)
        conditioning = cfgv.conditioning * replicate
        conditioning_local = conditioning[LRANK : len(conditioning) : LWSIZE]
        seed_increment = [i for i in range(len(conditioning))][LRANK : len(conditioning) : LWSIZE]

        for prompti, seedi in zip(conditioning_local, seed_increment):
            if prompti == 'SKIP':
                continue
            draw_filename = prompti.strip().replace(' ', '-')
            if fix_seed:
                np.random.seed(cfg.env.rnd_seed + seedi)
                torch.manual_seed(cfg.env.rnd_seed + seedi + 100)
                suffixi = suffix + "_seed{}".format(cfg.env.rnd_seed + seedi + 100)
            else:
                suffixi = suffix

            if with_ema:
                with netm.ema_scope():
                    x, _ = self.sample(netm, sampler, prompti, **cfgv.sample)
            else:
                x, _ = self.sample(netm, sampler, prompti, **cfgv.sample)

            demo_image = latent2im(netm, x)
            self.save_images(demo_image, draw_filename, image_path, suffix=suffixi)

        if eval_cnt is not None:
            print_log('Demo printed for {}'.format(eval_cnt))
        return {}

##################
# eval variation #
##################

class eval_stage_variation(eval_stage):
    @torch.no_grad()
    def sample(self, net, sampler, visual_hint, output_dim, scale, n_samples, ddim_steps, ddim_eta):
        h, w = output_dim
        vh = tvtrans.ToTensor()(PIL.Image.open(visual_hint))[None].to(net.device)
        c = net.get_learned_conditioning(vh)
        c = c.repeat(n_samples, 1, 1)
        uc = None
        if scale != 1.0:
            dummy = torch.zeros_like(vh)
            uc = net.get_learned_conditioning(dummy)
            uc = uc.repeat(n_samples, 1, 1)

        shape = [4, h//8, w//8]
        rv = sampler.sample(
            S=ddim_steps,
            conditioning=c,
            batch_size=n_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta)
        return rv

    def __call__(self, **paras):
        cfg = cfguh().cfg
        cfgv = cfg.eval

        net = paras['net']
        eval_cnt = paras.get('eval_cnt', None)
        fix_seed = cfgv.get('fix_seed', False)

        LRANK = sync.get_rank('local')
        LWSIZE = sync.get_world_size('local')

        image_path = self.get_image_path()
        self.create_dir(image_path)
        eval_cnt = paras.get('eval_cnt', None)
        suffix='' if eval_cnt is None else '_'+str(eval_cnt)

        if isinstance(net, (torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel)):
            netm = net.module
        else:
            netm = net

        with_ema = getattr(netm, 'model_ema', None) is not None
        sampler = self.sampler(netm)
        setattr(netm, 'device', LRANK) # Trick

        color_adj = cfguh().cfg.eval.get('color_adj', False)
        color_adj_keep_ratio = cfguh().cfg.eval.get('color_adj_keep_ratio', 0.5)
        color_adj_simple = cfguh().cfg.eval.get('color_adj_simple', True)

        replicate = cfgv.get('replicate', 1)
        conditioning = cfgv.conditioning * replicate
        conditioning_local = conditioning[LRANK : len(conditioning) : LWSIZE]
        seed_increment = [i for i in range(len(conditioning))][LRANK : len(conditioning) : LWSIZE]

        for ci, seedi in zip(conditioning_local, seed_increment):
            if ci == 'SKIP':
                continue

            draw_filename = osp.splitext(osp.basename(ci))[0]

            if fix_seed:
                np.random.seed(cfg.env.rnd_seed + seedi)
                torch.manual_seed(cfg.env.rnd_seed + seedi + 100)
                suffixi = suffix + "_seed{}".format(cfg.env.rnd_seed + seedi + 100)
            else:
                suffixi = suffix

            if with_ema:
                with netm.ema_scope():
                    x, _ = self.sample(netm, sampler, ci, **cfgv.sample)
            else:
                x, _ = self.sample(netm, sampler, ci, **cfgv.sample)

            demo_image = latent2im(netm, x)
            if color_adj:
                x_adj = []
                for demoi in demo_image:
                    color_adj_f = color_adjust(ref_from=demoi, ref_to=ci)
                    xi_adj = color_adj_f(demoi, keep=color_adj_keep_ratio, simple=color_adj_simple)
                    x_adj.append(xi_adj)
                demo_image = x_adj
            self.save_images(demo_image, draw_filename, image_path, suffix=suffixi)

        if eval_cnt is not None:
            print_log('Demo printed for {}'.format(eval_cnt))
        return {}
