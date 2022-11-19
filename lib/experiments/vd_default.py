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

from .sd_default import auto_merge_imlist, latent2im, color_adjust
from .sd_default import eval as eval_base
from .sd_default import eval_stage as eval_stage_base

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
        if ki.find('autokl')!=0 and ki.find('optimus')!=0 and ki.find('clip')!=0]

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
            if ki.find('autokl')==0 or ki.find('optimus')==0 or ki.find('clip')==0]
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

###################
# official stages #
###################

class eval(eval_base):
    pass

class eval_stage(eval_stage_base):
    """
    Evaluation of both prompt and vision
    """
    def __init__(self):
        from ..model_zoo.ddim_vd import DDIMSampler_VD
        self.sampler = DDIMSampler_VD

    @torch.no_grad()
    def sample(
            self, net, sampler, context, otype, ctype, image_output_dim, text_latent_dim,
            scale, n_samples, ddim_steps, ddim_eta):
        if ctype == 'prompt':
            c = net.clip_encode_text(n_samples * [context])
            uc = None
            if scale != 1.0:
                uc = net.clip_encode_text(n_samples * [""])
        elif ctype == 'vision':
            context = context[None].repeat(n_samples, 1, 1, 1)
            c = net.clip_encode_vision(context)
            uc = None
            if scale != 1.0:
                dummy = torch.zeros_like(context)
                uc = net.clip_encode_vision(dummy)

        if otype == 'image':
            h, w = image_output_dim
            shape = [n_samples, 4, h//8, w//8]
            rv = sampler.sample(
                steps=ddim_steps,
                shape=shape,
                conditioning=c,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                xtype=otype, ctype=ctype,
                eta=ddim_eta,
                verbose=False,)
        elif otype == 'text':
            n = text_latent_dim
            shape = [n_samples, n]
            rv = sampler.sample(
                steps=ddim_steps,
                shape=shape,
                conditioning=c,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                xtype=otype, ctype=ctype,
                eta=ddim_eta,
                verbose=False,)

        return rv

    def decode_and_save(
            self, netm, z, xtype, ctype, path, name, suffix,
            color_adj=False, color_adj_to=None):
        if xtype == 'image':
            x = netm.autokl_decode(z)
            name = 't2i_'+name if ctype == 'prompt' else 'v2i_'+name
            if color_adj and (ctype=='vision'):
                keep_ratio = cfguh().cfg.eval.get('color_adj_keep_ratio', 0.5)
                simple = cfguh().cfg.eval.get('color_adj_simple', True)
                x_adj = []
                for xi in x:
                    color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
                    xi_adj = color_adj_f((xi+1)/2, keep=keep_ratio, simple=simple)
                    x_adj.append(xi_adj)
                x = x_adj
            self.save_images(x, name, path, suffix=suffix)
        elif xtype == 'text':
            prompt_temperature = cfguh().cfg.eval.get('prompt_temperature', 1.0)
            x = netm.optimus_decode(z, temperature=prompt_temperature)
            name = 't2t_'+name if ctype == 'prompt' else 'v2t_'+name
            prompt_merge_same_adj_word = cfguh().cfg.eval.get('prompt_merge_same_adj_word', False)
            if prompt_merge_same_adj_word:
                xnew = []
                for xi in x:
                    xi_split = xi.split()
                    xinew = []
                    for idxi, wi in enumerate(xi_split):
                        if idxi!=0 and wi==xi_split[idxi-1]:
                            continue
                        xinew.append(wi)
                    xnew.append(' '.join(xinew))
                x = xnew
            self.save_text(x, name, path, suffix=suffix)

    def save_images(self, x, name, path, suffix=''):
        if isinstance(x, torch.Tensor):
            single_input = len(x.shape) == 3
            if single_input:
                x = x[None]
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
            x = [tvtrans.ToPILImage()(xi) for xi in x]
            xlist = [np.array(xi) for xi in x]
        elif isinstance(x, list):
            xlist = x
        canvas = auto_merge_imlist(xlist)
        image_name = '{}{}.png'.format(name, suffix)
        PIL.Image.fromarray(canvas).save(osp.join(path, image_name))

    def save_text(self, x, name, path, suffix=''):
        file_name = '{}{}.txt'.format(name, suffix)
        with open(osp.join(path, file_name) ,'w') as f:
            for xi in x:
                f.write(xi+'\n')

    def __call__(self, **paras):
        cfg = cfguh().cfg
        cfgv = cfg.eval

        net = self.get_net(paras)
        eval_cnt = paras.get('eval_cnt', None)
        fix_seed = cfgv.get('fix_seed', False)

        LRANK = sync.get_rank('local')
        LWSIZE = sync.get_world_size('local')

        output_path = self.get_image_path()
        self.create_dir(output_path)
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

        replicate = cfgv.get('replicate', 1)
        conditioning = cfgv.conditioning * replicate
        conditioning_local = conditioning[LRANK : len(conditioning) : LWSIZE]
        seed_increment = [i for i in range(len(conditioning))][LRANK : len(conditioning) : LWSIZE]

        for conditioningi, seedi in zip(conditioning_local, seed_increment):
            if conditioningi == 'SKIP':
                continue

            ci, otypei = conditioningi

            if osp.isfile(ci):
                # is vision
                output_name = osp.splitext(osp.basename(ci))[0]
                ci = tvtrans.ToTensor()(PIL.Image.open(ci))
                ci = ci*2 - 1
                ctypei = 'vision'
            else:
                # is prompt
                output_name = ci.strip().replace(' ', '-')
                ctypei = 'prompt'

            if fix_seed:
                np.random.seed(cfg.env.rnd_seed + seedi)
                torch.manual_seed(cfg.env.rnd_seed + seedi + 100)
                suffixi = suffix + "_seed{}".format(cfg.env.rnd_seed + seedi + 100)
            else:
                suffixi = suffix

            if with_ema:
                with netm.ema_scope():
                    z, _ = self.sample(netm, sampler, ci, otypei, ctypei, **cfgv.sample)
            else:
                z, _ = self.sample(netm, sampler, ci, otypei, ctypei, **cfgv.sample)

            self.decode_and_save(
                netm, z, otypei, ctypei, output_path, output_name, suffixi,
                color_adj=color_adj, color_adj_to=conditioningi[0],)

        if eval_cnt is not None:
            print_log('Demo printed for {}'.format(eval_cnt))
        return {}

################
# basic stages #
################

class eval_stage_basic(eval_stage_base):
    @torch.no_grad()
    def sample(self, net, sampler, visual_hint, output_dim, scale, n_samples, ddim_steps, ddim_eta):
        h, w = output_dim
        vh = PIL.Image.open(visual_hint)
        c = net.clip_encode_vision(n_samples * [vh])
        uc = None
        if scale != 1.0:
            dummy = torch.zeros_like(tvtrans.ToTensor()(vh))
            uc = net.clip_encode_vision(n_samples * [dummy])

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

#######################
# dual context stages #
#######################

class eval_stage_dc(eval_stage_base):
    def __init__(self):
        from ..model_zoo.ddim_dualcontext import DDIMSampler_DualContext
        self.sampler = DDIMSampler_DualContext

    @torch.no_grad()
    def sample(
            self, net, sampler, conditioning, output_dim, 
            scale, n_samples, ddim_steps, ddim_eta):
        ctype, cvalue =conditioning
        if ctype == 'prompt':
            return self.sample_text(
                net, sampler, cvalue, output_dim,
                scale, n_samples, ddim_steps, ddim_eta)
        elif ctype == 'vision':
            return self.sample_vision(
                net, sampler, cvalue, output_dim,
                scale, n_samples, ddim_steps, ddim_eta)
        else:
            raise ValueError

    @torch.no_grad()
    def sample_text(
            self, net, sampler, prompt, output_dim, 
            scale, n_samples, ddim_steps, ddim_eta):
        h, w = output_dim
        uc = None
        if scale != 1.0:
            uc = net.clip_encode_text(n_samples * [""])
        c = net.clip_encode_text(n_samples * [prompt])
        shape = [n_samples, 4, h//8, w//8]
        rv = sampler.sample_text(
            steps=ddim_steps,
            shape=shape,
            conditioning=c,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            verbose=False,)
        return rv

    @torch.no_grad()
    def sample_vision(
            self, net, sampler, visual_hint, output_dim, 
            scale, n_samples, ddim_steps, ddim_eta):
        h, w = output_dim
        if len(visual_hint.shape) == 3:
            visual_hint=visual_hint[None].repeat(n_samples, 1, 1, 1)
        else:
            raise ValueError

        c = net.clip_encode_vision(visual_hint)
        uc = None
        if scale != 1.0:
            visual_hint_blank = torch.zeros_like(visual_hint)
            uc = net.clip_encode_vision(visual_hint_blank)

        shape = [n_samples, 4, h//8, w//8]
        rv = sampler.sample_vision(
            steps=ddim_steps,
            shape=shape,
            conditioning=c,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            verbose=False,)
        return rv

    def __call__(self, **paras):
        cfg = cfguh().cfg
        cfgv = cfg.eval

        net = self.get_net(paras)
        eval_cnt = paras.get('eval_cnt', None)
        fix_seed = cfgv.get('fix_seed', False)

        LRANK = sync.get_rank('local')
        LWSIZE = sync.get_world_size('local')

        image_path = self.get_image_path()
        self.create_dir(image_path)
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

            if osp.isfile(ci):
                # is vision
                draw_filename = 'v2i_' + osp.splitext(osp.basename(ci))[0]
                ci = tvtrans.ToTensor()(PIL.Image.open(ci))
                ci = ci*2 - 1
                ci = ('vision', ci)
            else:
                # is prompt
                draw_filename = 't2i_' + ci.strip().replace(' ', '-')
                ci = ('prompt', ci)

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
            if color_adj and ci[0] == 'vision':
                x_adj = []
                for demoi in demo_image:
                    color_adj_f = color_adjust(ref_from=demoi, ref_to=ci[1])
                    xi_adj = color_adj_f(demoi, keep=color_adj_keep_ratio, simple=color_adj_simple)
                    x_adj.append(xi_adj)
                demo_image = x_adj
            self.save_images(demo_image, draw_filename, image_path, suffix=suffixi)

        if eval_cnt is not None:
            print_log('Demo printed for {}'.format(eval_cnt))
        return {}

