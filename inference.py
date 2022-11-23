import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_dualcontext import DDIMSampler_DualContext
from lib.experiments.sd_default import color_adjust, auto_merge_imlist

import argparse

n_sample_image_default = 2
n_sample_text_default = 4

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

class vd_inference(object):
    def __init__(self, pth='pretrained/vd1.0-four-flow.pth', fp16=False, device=0):
        cfgm_name = 'vd_noema'
        cfgm = model_cfg_bank()('vd_noema')
        device_str = device if isinstance(device, str) else 'cuda:{}'.format(device)
        cfgm.args.autokl_cfg.map_location = device_str
        cfgm.args.optimus_cfg.map_location = device_str
        net = get_model()(cfgm)
        if fp16:
            highlight_print('Running in FP16')
            net.clip.fp16 = True
            net = net.half()
        sd = torch.load(pth, map_location=device_str)
        net.load_state_dict(sd, strict=False)
        print('Load pretrained weight from {}'.format(pth))
        net.to(device)

        self.device = device
        self.model_name = cfgm_name
        self.net = net
        self.fp16 = fp16
        from lib.model_zoo.ddim_vd import DDIMSampler_VD
        self.sampler = DDIMSampler_VD(net)

    def regularize_image(self, x):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        x = x.to(self.device)
        if self.fp16:
            x = x.half()
        return x

    def decode(self, z, xtype, ctype, color_adj='None', color_adj_to=None):
        net = self.net
        if xtype == 'image':
            x = net.autokl_decode(z)

            color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
            color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
            color_adj_keep_ratio = 0.5

            if color_adj_flag and (ctype=='vision'):
                x_adj = []
                for xi in x:
                    color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
                    xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
                    x_adj.append(xi_adj)
                x = x_adj
            else:
                x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
                x = [tvtrans.ToPILImage()(xi) for xi in x]
            return x

        elif xtype == 'text':
            prompt_temperature = 1.0
            prompt_merge_same_adj_word = True
            x = net.optimus_decode(z, temperature=prompt_temperature)
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
            return x

    def inference(self, xtype, cin, ctype, scale=7.5, n_samples=None, color_adj=None,):
        net = self.net
        sampler = self.sampler
        ddim_steps = 50
        ddim_eta = 0.0

        if xtype == 'image':
            n_samples = n_sample_image_default if n_samples is None else n_samples
        elif xtype == 'text':
            n_samples = n_sample_text_default if n_samples is None else n_samples

        if ctype in ['prompt', 'text']:
            c = net.clip_encode_text(n_samples * [cin])
            u = None
            if scale != 1.0:
                u = net.clip_encode_text(n_samples * [""])

        elif ctype in ['vision', 'image']:
            cin = self.regularize_image(cin)
            ctemp = cin*2 - 1
            ctemp = ctemp[None].repeat(n_samples, 1, 1, 1)
            c = net.clip_encode_vision(ctemp)
            u = None
            if scale != 1.0:
                dummy = torch.zeros_like(ctemp)
                u = net.clip_encode_vision(dummy)

        u, c = [u.half(), c.half()] if self.fp16 else [u, c]

        if xtype == 'image':
            h, w = [512, 512]
            shape = [n_samples, 4, h//8, w//8]
            z, _ = sampler.sample(
                steps=ddim_steps,
                shape=shape,
                conditioning=c,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=u,
                xtype=xtype, ctype=ctype,
                eta=ddim_eta,
                verbose=False,)
            x = self.decode(z, xtype, ctype, color_adj=color_adj, color_adj_to=cin)
            return x

        elif xtype == 'text':
            n = 768
            shape = [n_samples, n]
            z, _ = sampler.sample(
                steps=ddim_steps,
                shape=shape,
                conditioning=c,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=u,
                xtype=xtype, ctype=ctype,
                eta=ddim_eta,
                verbose=False,)
            x = self.decode(z, xtype, ctype)
            return x

    def application_disensemble(self, cin, n_samples=None, level=0, color_adj=None,):
        net = self.net
        scale = 7.5
        sampler = self.sampler
        ddim_steps = 50
        ddim_eta = 0.0
        n_samples = n_sample_image_default if n_samples is None else n_samples

        cin = self.regularize_image(cin)
        ctemp = cin*2 - 1
        ctemp = ctemp[None].repeat(n_samples, 1, 1, 1)
        c = net.clip_encode_vision(ctemp)
        u = None
        if scale != 1.0:
            dummy = torch.zeros_like(ctemp)
            u = net.clip_encode_vision(dummy)
        u, c = [u.half(), c.half()] if self.fp16 else [u, c]

        if level == 0:
            pass
        else:
            c_glb = c[:, 0:1]
            c_loc = c[:, 1: ]
            u_glb = u[:, 0:1]
            u_loc = u[:, 1: ]

            if level == -1:
                c_loc = self.remove_low_rank(c_loc, demean=True, q=50, q_remove=1)
                u_loc = self.remove_low_rank(u_loc, demean=True, q=50, q_remove=1)
            if level == -2:
                c_loc = self.remove_low_rank(c_loc, demean=True, q=50, q_remove=2)
                u_loc = self.remove_low_rank(u_loc, demean=True, q=50, q_remove=2)
            if level == 1:
                c_loc = self.find_low_rank(c_loc, demean=True, q=10)
                u_loc = self.find_low_rank(u_loc, demean=True, q=10)
            if level == 2:
                c_loc = self.find_low_rank(c_loc, demean=True, q=2)
                u_loc = self.find_low_rank(u_loc, demean=True, q=2)

            c = torch.cat([c_glb, c_loc], dim=1)
            u = torch.cat([u_glb, u_loc], dim=1)

        h, w = [512, 512]
        shape = [n_samples, 4, h//8, w//8]
        z, _ = sampler.sample(
            steps=ddim_steps,
            shape=shape,
            conditioning=c,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=u,
            xtype='image', ctype='vision',
            eta=ddim_eta,
            verbose=False,)
        x = self.decode(z, 'image', 'vision', color_adj=color_adj, color_adj_to=cin)
        return x

    def find_low_rank(self, x, demean=True, q=20, niter=10):
        if demean:
            x_mean = x.mean(-1, keepdim=True)
            x_input = x - x_mean
        else:
            x_input = x

        if x_input.dtype == torch.float16:
            fp16 = True
            x_input = x_input.float()
        else:
            fp16 = False

        u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))        

        if fp16:
            x_lowrank = x_lowrank.half()

        if demean:
            x_lowrank += x_mean
        return x_lowrank

    def remove_low_rank(self, x, demean=True, q=20, niter=10, q_remove=10):
        if demean:
            x_mean = x.mean(-1, keepdim=True)
            x_input = x - x_mean
        else:
            x_input = x

        if x_input.dtype == torch.float16:
            fp16 = True
            x_input = x_input.float()
        else:
            fp16 = False

        u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
        s[:, 0:q_remove] = 0
        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))        

        if fp16:
            x_lowrank = x_lowrank.half()

        if demean:
            x_lowrank += x_mean
        return x_lowrank

    def application_dualguided(self, cim, ctx, n_samples=None, mixing=0.5, color_adj=None, ):
        net = self.net
        scale = 7.5
        sampler = self.sampler
        ddim_steps = 50
        ddim_eta = 0.0
        n_samples = n_sample_image_default if n_samples is None else n_samples

        ctemp0 = self.regularize_image(cim)
        ctemp1 = ctemp0*2 - 1
        ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1)
        cim = net.clip_encode_vision(ctemp1)
        uim = None
        if scale != 1.0:
            dummy = torch.zeros_like(ctemp1)
            uim = net.clip_encode_vision(dummy)

        ctx = net.clip_encode_text(n_samples * [ctx])
        utx = None
        if scale != 1.0:
            utx = net.clip_encode_text(n_samples * [""])

        uim, cim = [uim.half(), cim.half()] if self.fp16 else [uim, cim]
        utx, ctx = [utx.half(), ctx.half()] if self.fp16 else [utx, ctx]

        h, w = [512, 512]
        shape = [n_samples, 4, h//8, w//8]

        z, _ = sampler.sample_dc(
            steps=ddim_steps,
            shape=shape,
            first_conditioning=[uim, cim],
            second_conditioning=[utx, ctx],
            unconditional_guidance_scale=scale,
            xtype='image', 
            first_ctype='vision',
            second_ctype='prompt',
            eta=ddim_eta,
            verbose=False,
            mixed_ratio=(1-mixing), )
        x = self.decode(z, 'image', 'vision', color_adj=color_adj, color_adj_to=ctemp0)
        return x

    def application_i2t2i(self, cim, ctx_n, ctx_p, n_samples=None, color_adj=None,):
        net = self.net
        scale = 7.5
        sampler = self.sampler
        ddim_steps = 50
        ddim_eta = 0.0
        prompt_temperature = 1.0
        n_samples = n_sample_image_default if n_samples is None else n_samples

        ctemp0 = self.regularize_image(cim)
        ctemp1 = ctemp0*2 - 1
        ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1)
        cim = net.clip_encode_vision(ctemp1)
        uim = None
        if scale != 1.0:
            dummy = torch.zeros_like(ctemp1)
            uim = net.clip_encode_vision(dummy)

        uim, cim = [uim.half(), cim.half()] if self.fp16 else [uim, cim]

        n = 768
        shape = [n_samples, n]
        zt, _ = sampler.sample(
            steps=ddim_steps,
            shape=shape,
            conditioning=cim,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uim,
            xtype='text', ctype='vision',
            eta=ddim_eta,
            verbose=False,)
        ztn = net.optimus_encode([ctx_n])
        ztp = net.optimus_encode([ctx_p])

        ztn_norm = ztn / ztn.norm(dim=1)
        zt_proj_mag = torch.matmul(zt, ztn_norm[0])
        zt_perp = zt - zt_proj_mag[:, None] * ztn_norm
        zt_newd = zt_perp + ztp
        ctx_new = net.optimus_decode(zt_newd, temperature=prompt_temperature)

        ctx_new = net.clip_encode_text(ctx_new)
        ctx_p = net.clip_encode_text([ctx_p])
        ctx_new = torch.cat([ctx_new, ctx_p.repeat(n_samples, 1, 1)], dim=1)
        utx_new = net.clip_encode_text(n_samples * [""])
        utx_new = torch.cat([utx_new, utx_new], dim=1)

        cim_loc = cim[:, 1: ]
        cim_loc_new = self.find_low_rank(cim_loc, demean=True, q=10)
        cim_new = cim_loc_new
        uim_new = uim[:, 1:]
        
        h, w = [512, 512]
        shape = [n_samples, 4, h//8, w//8]
        z, _ = sampler.sample_dc(
            steps=ddim_steps,
            shape=shape,
            first_conditioning=[uim_new, cim_new],
            second_conditioning=[utx_new, ctx_new],
            unconditional_guidance_scale=scale,
            xtype='image', 
            first_ctype='vision',
            second_ctype='prompt',
            eta=ddim_eta,
            verbose=False,
            mixed_ratio=0.33, )

        x = self.decode(z, 'image', 'vision', color_adj=color_adj, color_adj_to=ctemp0)
        return x

def main(netwrapper,
         app,
         image=None,
         prompt=None,
         nprompt=None,
         pprompt=None,
         color_adj=None,
         disentanglement_level=None,
         dual_guided_mixing=None,
         n_samples=4,
         seed=0,):

    if seed is not None:
        seed = 0 if seed<0 else seed
        np.random.seed(seed)
        torch.manual_seed(seed+100)

    if app == 'text-to-image':
        print('Running [{}] with prompt [{}], n_samples [{}], seed [{}].'.format(
            app, prompt, n_samples, seed))
        if (prompt is None) or (prompt == ""):
            return None, None
        with torch.no_grad():
            rv = netwrapper.inference(
                xtype = 'image',
                cin = prompt,
                ctype = 'prompt', 
                n_samples = n_samples, )
        return rv, None

    elif app == 'image-variation':
        print('Running [{}] with image [{}], color_adj [{}], n_samples [{}], seed [{}].'.format(
            app, image, color_adj, n_samples, seed))
        if image is None:
            return None, None
        with torch.no_grad():
            rv = netwrapper.inference(
                xtype = 'image',
                cin = image,
                ctype = 'vision',
                color_adj = color_adj,
                n_samples = n_samples, )
        return rv, None

    elif app == 'image-to-text':
        print('Running [{}] with iamge [{}], n_samples [{}], seed [{}].'.format(
            app, image, n_samples, seed))
        if image is None:
            return None, None
        with torch.no_grad():
            rv = netwrapper.inference(
                xtype = 'text',
                cin = image,
                ctype = 'vision',
                n_samples = n_samples, )
        return None, '\n'.join(rv)

    elif app == 'text-variation':
        print('Running [{}] with prompt [{}], n_samples [{}], seed [{}].'.format(
            app, prompt, n_samples, seed))
        if prompt is None:
            return None, None
        with torch.no_grad():
            rv = netwrapper.inference(
                xtype = 'text',
                cin = prompt,
                ctype = 'prompt',
                n_samples = n_samples, )
        return None, '\n'.join(rv)

    elif app == 'disentanglement':
        print('Running [{}] with image [{}], color_adj [{}], disentanglement_level [{}], n_samples [{}], seed [{}].'.format(
            app, image, color_adj, disentanglement_level, n_samples, seed))
        if image is None:
            return None, None
        with torch.no_grad():
            rv = netwrapper.application_disensemble(
                cin = image,
                level = disentanglement_level,
                color_adj = color_adj,
                n_samples = n_samples, )
        return rv, None

    elif app == 'dual-guided':
        print('Running [{}] with image [{}], prompt [{}], color_adj [{}], dual_guided_mixing [{}], n_samples [{}], seed [{}].'.format(
            app, image, prompt, color_adj, dual_guided_mixing, n_samples, seed))
        if (image is None) or (prompt is None) or (prompt==""):
            return None, None
        with torch.no_grad():
            rv = netwrapper.application_dualguided(
                cim = image,
                ctx = prompt,
                mixing = dual_guided_mixing,
                color_adj = color_adj,
                n_samples = n_samples, )
        return rv, None

    elif app == 'i2t2i':
        print('Running [{}] with image [{}], nprompt [{}], pprompt [{}], color_adj [{}], n_samples [{}], seed [{}].'.format(
            app, image, nprompt, pprompt, color_adj, n_samples, seed))
        if (image is None) or (nprompt is None) or (nprompt=="") \
                or (pprompt is None) or (pprompt==""):
            return None, None
        with torch.no_grad():
            rv = netwrapper.application_i2t2i(
                cim = image,
                ctx_n = nprompt,
                ctx_p = pprompt,
                color_adj = color_adj,
                n_samples = n_samples, )
        return rv, None
    
    else:
        assert False, "No such mode!"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app", type=str, default="text-to-image",
        help="Choose the application from ["\
             "text-to-image, image-variation, "\
             "image-to-text, text-variation, "\
             "disentanglement, dual-guided, i2t2i]")

    parser.add_argument(
        "--model", type=str, default="official",
        help="Choose the model type from ["\
             "dc, official]")

    parser.add_argument(
        "--prompt", type=str, 
        default="a dream of a village in china, by Caspar "\
                "David Friedrich, matte painting trending on artstation HQ")

    parser.add_argument("--image", type=str)

    parser.add_argument("--nprompt", type=str)

    parser.add_argument("--pprompt", type=str)

    parser.add_argument("--coloradj", type=str, default='simple')

    parser.add_argument("--dislevel", type=int, default=0)

    parser.add_argument("--dgmixing", type=float, default=0.7)

    parser.add_argument("--nsample", type=int, default=4)

    parser.add_argument("--seed", type=int)

    parser.add_argument("--save", type=str, default='log',
        help="The path or file the result will save into")

    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--fp16", action="store_true")

    # parser.add_argument("--pth", type=str, default='pretrained/vd-four-flow-v1-0.pth')

    args = parser.parse_args()

    assert args.app in [
            "text-to-image", "image-variation",
            "image-to-text", "text-variation",
            "disentanglement", "dual-guided", "i2t2i"], \
        "Unknown app! Select from [text-to-image, image-variation, "\
        "image-to-text, text-variation, "\
        "disentanglement, dual-guided, i2t2i]"

    device=args.gpu if torch.cuda.is_available() else 'cpu'

    if args.model in ['4-flow', 'official']:
        if args.fp16:
            pth='pretrained/vd-four-flow-v1-0-fp16.pth'
        else:
            pth='pretrained/vd-four-flow-v1-0.pth'
        vd_wrapper = vd_inference(pth=pth, fp16=args.fp16, device=device)
    elif args.model in ['2-flow', 'dc']:
        raise NotImplementedError
        # vd_wrapper = vd_dc_inference(args.model, pth=args.pth, device=device)
    elif args.model in ['1-flow', 'basic']:
        raise NotImplementedError
        # vd_wrapper = vd_basic_inference(args.model, pth=args.pth, device=device)
    else:
        assert False, "No such model! Select model from [4-flow(official), 2-flow(dc), 1-flow(basic)]"

    imout, txtout = main(
        netwrapper=vd_wrapper,
        app=args.app,
        image=args.image,
        prompt=args.prompt,
        nprompt=args.nprompt,
        pprompt=args.pprompt,
        color_adj=args.coloradj,
        disentanglement_level=args.dislevel,
        dual_guided_mixing=args.dgmixing,
        n_samples=args.nsample,
        seed=args.seed,)

    if imout is not None:
        imout = auto_merge_imlist([np.array(i) for i in imout])
        imout = PIL.Image.fromarray(imout)
        if osp.isdir(args.save):
            imout.save(osp.join(args.save, 'imout.png'))
            print('Output image saved to {}.'.format(osp.join(args.save, 'imout.png')))
        else:
            imout.save(osp.join(args.save))
            print('Output image saved to {}.'.format(args.save))
    
    if txtout is not None:
        print(txtout)
