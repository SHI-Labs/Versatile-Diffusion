################################################################################
# Copyright (C) 2023 Xingqian Xu - All Rights Reserved                         #
#                                                                              #
# Please visit Versatile Diffusion's arXiv paper for more details, link at     #
# arxiv.org/abs/2211.08332                                                     #
#                                                                              #
# Besides, this work is also inspired by many established techniques including:#
# Denoising Diffusion Probablistic Model; Denoising Diffusion Implicit Model;  #
# Latent Diffusion Model; Stable Diffusion; Stable Diffusion - Img2Img; Stable #
# Diffusion - Variation; ImageMixer; DreamBooth; Stable Diffusion - Lora; More #
# Control for Free; Prompt-to-Prompt;                                          #
#                                                                              #
################################################################################

import gradio as gr
import os
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr
from contextlib import nullcontext
import types

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from cusomized_gradio_blocks import create_myexamples, customized_as_example, customized_postprocess

n_sample_image = 2
n_sample_text = 4
cache_examples = True

from lib.model_zoo.ddim import DDIMSampler

##########
# helper #
##########

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

def decompose(x, q=20, niter=100):
    x_mean = x.mean(-1, keepdim=True)
    x_input = x - x_mean
    u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
    ss = torch.stack([torch.diag(si) for si in s])
    x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
    x_remain = x_input - x_lowrank
    return u, s, v, x_mean, x_remain

class adjust_rank(object):
    def __init__(self, max_drop_rank=[1, 5], q=20):
        self.max_semantic_drop_rank = max_drop_rank[0]
        self.max_style_drop_rank = max_drop_rank[1]
        self.q = q

        def t2y0_semf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((0  -0.5)*2), -self.max_semantic_drop_rank
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_semf = t2y0_semf_wrapper(t0, y00, t1, y01)

        def x2y_semf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = 0
        x1, y1 = self.max_semantic_drop_rank+1, 1
        self.x2y_semf = x2y_semf_wrapper(x0, x1, y1)
        
        def t2y0_styf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((1  -0.5)*2), -(q-self.max_style_drop_rank)
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_styf = t2y0_styf_wrapper(t0, y00, t1, y01)

        def x2y_styf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = q-1
        x1, y1 = self.max_style_drop_rank-1, 1
        self.x2y_styf = x2y_styf_wrapper(x0, x1, y1)

    def __call__(self, x, lvl):
        if lvl == 0.5:
            return x

        if x.dtype == torch.float16:
            fp16 = True
            x = x.float()
        else:
            fp16 = False
        std_save = x.std(axis=[-2, -1])

        u, s, v, x_mean, x_remain = decompose(x, q=self.q)

        if lvl < 0.5:
            assert lvl>=0
            for xi in range(0, self.max_semantic_drop_rank+1):
                y0 = self.t2y0_semf(lvl)
                yi = self.x2y_semf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi

        elif lvl > 0.5:
            assert lvl <= 1
            for xi in range(self.max_style_drop_rank, self.q):
                y0 = self.t2y0_styf(lvl)
                yi = self.x2y_styf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi
            x_remain = 0

        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
        x_new = x_lowrank + x_mean + x_remain

        std_new = x_new.std(axis=[-2, -1])
        x_new = x_new / std_new * std_save

        if fp16:
            x_new = x_new.half()

        return x_new

def remove_duplicate_word(tx):
    def combine_words(input, length):
        combined_inputs = []
        if len(splitted_input)>1:
            for i in range(len(input)-1):
                combined_inputs.append(input[i]+" "+last_word_of(splitted_input[i+1],length)) #add the last word of the right-neighbour (overlapping) sequence (before it has expanded), which is the next word in the original sentence
        return combined_inputs, length+1

    def remove_duplicates(input, length):
        bool_broke=False #this means we didn't find any duplicates here
        for i in range(len(input) - length):
            if input[i]==input[i + length]: #found a duplicate piece of sentence!
                for j in range(0, length): #remove the overlapping sequences in reverse order
                    del input[i + length - j]
                bool_broke = True
                break #break the for loop as the loop length does not matches the length of splitted_input anymore as we removed elements
        if bool_broke:
            return remove_duplicates(input, length) #if we found a duplicate, look for another duplicate of the same length
        return input

    def last_word_of(input, length):
        splitted = input.split(" ")
        if len(splitted)==0:
            return input
        else:
            return splitted[length-1]

    def split_and_puncsplit(text):
        tx = text.split(" ")
        txnew = []
        for txi in tx:
            txqueue=[]
            while True:
                if txi[0] in '([{':
                    txqueue.extend([txi[:1], '<puncnext>'])
                    txi = txi[1:]
                    if len(txi) == 0:
                        break
                else:
                    break
            txnew += txqueue
            txstack=[]
            if len(txi) == 0:
                continue
            while True:
                if txi[-1] in '?!.,:;}])':
                    txstack = ['<puncnext>', txi[-1:]] + txstack
                    txi = txi[:-1]
                    if len(txi) == 0:
                        break
                else:
                    break
            if len(txi) != 0:
                txnew += [txi]
            txnew += txstack
        return txnew

    if tx == '':
        return tx

    splitted_input = split_and_puncsplit(tx)
    word_length = 1
    intermediate_output = False
    while len(splitted_input)>1:
        splitted_input = remove_duplicates(splitted_input, word_length)
        if len(splitted_input)>1:
            splitted_input, word_length = combine_words(splitted_input, word_length)
        if intermediate_output:
            print(splitted_input)
            print(word_length)
    output = splitted_input[0]
    output = output.replace(' <puncnext> ', '')
    return output

def get_instruction(mode):
    t2i_instruction = ["Generate image from text prompt."]
    i2i_instruction = ["Generate image conditioned on reference image.",]
    i2t_instruction = ["Generate text from reference image. "]
    t2t_instruction = ["Generate text from reference text prompt. "]
    dcg_instruction = ["Generate image conditioned on both text and image."]
    tcg_instruction = ["Generate image conditioned on text and up to two images."]
    mcg_instruction = ["Generate image from multiple contexts."]

    if mode == "Text-to-Image":
        return '\n'.join(t2i_instruction)
    elif mode == "Image-Variation":
        return '\n'.join(i2i_instruction)
    elif mode == "Image-to-Text":
        return '\n'.join(i2t_instruction)
    elif mode == "Text-Variation":
        return '\n'.join(t2t_instruction)
    elif mode == "Dual-Context":
        return '\n'.join(dcg_instruction)
    elif mode == "Triple-Context":
        return '\n'.join(tcg_instruction)
    elif mode == "Multi-Context":
        return '\n'.join(mcg_instruction)
    else:
        assert False

########
# main #
########
class vd_dummy(object):
    def __init__(self, *args, **kwarg):
        self.which = 'Vdummy'
    def inference_t2i(self, *args, **kwarg): pass
    def inference_i2i(self, *args, **kwarg): pass
    def inference_i2t(self, *args, **kwarg): pass
    def inference_t2t(self, *args, **kwarg): pass
    def inference_dcg(self, *args, **kwarg): pass
    def inference_tcg(self, *args, **kwarg): pass
    def inference_mcg(self, *args, **kwarg): 
        return None, None

class vd_inference(object):
    def __init__(self, fp16=False, which='v2.0'):
        highlight_print(which)
        self.which = which

        if self.which == 'v1.0':
            cfgm = model_cfg_bank()('vd_four_flow_v1-0')
        else:
            assert False, 'Model type not supported'
        net = get_model()(cfgm)

        if fp16:
            highlight_print('Running in FP16')
            if self.which == 'v1.0':
                net.ctx['text'].fp16 = True
                net.ctx['image'].fp16 = True
            net = net.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        if self.which == 'v1.0':
            if fp16:
                sd = torch.load('pretrained/vd-four-flow-v1-0-fp16.pth', map_location='cpu')
            else:
                sd = torch.load('pretrained/vd-four-flow-v1-0.pth', map_location='cpu')
            # from huggingface_hub import hf_hub_download
            # if fp16:
            #     temppath = hf_hub_download('shi-labs/versatile-diffusion-model', 'pretrained_pth/vd-four-flow-v1-0-fp16.pth')
            # else:
            #     temppath = hf_hub_download('shi-labs/versatile-diffusion-model', 'pretrained_pth/vd-four-flow-v1-0.pth')
            # sd = torch.load(temppath, map_location='cpu')

        net.load_state_dict(sd, strict=False)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            net.to('cuda')
        self.net = net
        self.sampler = DDIMSampler(net)

        self.output_dim = [512, 512]
        self.n_sample_image = n_sample_image
        self.n_sample_text = n_sample_text
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.scale_textto = 7.5
        self.image_latent_dim = 4
        self.text_latent_dim = 768
        self.text_temperature = 1

        if which == 'v1.0':
            self.adjust_rank_f = adjust_rank(max_drop_rank=[1, 5], q=20)
            self.scale_imgto = 7.5
            self.disentanglement_noglobal = True

    def inference_t2i(self, text, seed):
        n_samples = self.n_sample_image
        scale = self.scale_textto
        sampler = self.sampler
        h, w = self.output_dim
        u = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
        c = self.net.ctx_encode([text], which='text').repeat(n_samples, 1, 1)
        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type':'image'},
            c_info={'type':'text', 'conditioning':c, 'unconditional_conditioning':u, 
                    'unconditional_guidance_scale':scale},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)
        im = self.net.vae_decode(x, which='image')
        im = [tvtrans.ToPILImage()(i) for i in im]
        return im

    def inference_i2i(self, im, fid_lvl, fcs_lvl, clr_adj, seed):
        n_samples = self.n_sample_image
        scale = self.scale_imgto
        sampler = self.sampler
        h, w = self.output_dim
        device = self.net.device

        BICUBIC = PIL.Image.Resampling.BICUBIC
        im = im.resize([w, h], resample=BICUBIC)

        if fid_lvl == 1:
            return [im]*n_samples

        cx = tvtrans.ToTensor()(im)[None].to(device).to(self.dtype)

        c = self.net.ctx_encode(cx, which='image')
        if self.disentanglement_noglobal:
            c_glb = c[:, 0:1]
            c_loc = c[:, 1: ]
            c_loc = self.adjust_rank_f(c_loc, fcs_lvl)
            c = torch.cat([c_glb, c_loc], dim=1).repeat(n_samples, 1, 1)
        else:
            c = self.adjust_rank_f(c, fcs_lvl).repeat(n_samples, 1, 1)
        u = torch.zeros_like(c)

        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        if fid_lvl!=0:
            x0 = self.net.vae_encode(cx, which='image').repeat(n_samples, 1, 1, 1)
            step = int(self.ddim_steps * (1-fid_lvl))
            x, _ = sampler.sample(
                steps=self.ddim_steps,
                x_info={'type':'image', 'x0':x0, 'x0_forward_timesteps':step},
                c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                        'unconditional_guidance_scale':scale},
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)
        else:
            x, _ = sampler.sample(
                steps=self.ddim_steps,
                x_info={'type':'image',},
                c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                        'unconditional_guidance_scale':scale},
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)

        imout = self.net.vae_decode(x, which='image')

        if clr_adj == 'Simple':
            cx_mean = cx.view(3, -1).mean(-1)[:, None, None]
            cx_std  = cx.view(3, -1).std(-1)[:, None, None]
            imout_mean = [imouti.view(3, -1).mean(-1)[:, None, None] for imouti in imout]
            imout_std  = [imouti.view(3, -1).std(-1)[:, None, None] for imouti in imout]
            imout = [(ii-mi)/si*cx_std+cx_mean for ii, mi, si in zip(imout, imout_mean, imout_std)]
            imout = [torch.clamp(ii, 0, 1) for ii in imout]

        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return imout

    def inference_i2t(self, im, seed):
        n_samples = self.n_sample_text
        scale = self.scale_imgto
        sampler = self.sampler
        h, w = self.output_dim
        device = self.net.device

        BICUBIC = PIL.Image.Resampling.BICUBIC
        im = im.resize([w, h], resample=BICUBIC)

        cx = tvtrans.ToTensor()(im)[None].to(device)
        c = self.net.ctx_encode(cx, which='image').repeat(n_samples, 1, 1)
        u = self.net.ctx_encode(torch.zeros_like(cx), which='image').repeat(n_samples, 1, 1)

        shape = [n_samples, self.text_latent_dim]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type':'text',},
            c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                    'unconditional_guidance_scale':scale},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)
        tx = self.net.vae_decode(x, which='text', temperature=self.text_temperature)
        tx = [remove_duplicate_word(txi) for txi in tx]
        tx_combined = '\n'.join(tx)
        return tx_combined

    def inference_t2t(self, text, seed):
        n_samples = self.n_sample_text
        scale = self.scale_textto
        sampler = self.sampler
        u = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
        c = self.net.ctx_encode([text], which='text').repeat(n_samples, 1, 1)
        shape = [n_samples, self.text_latent_dim]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type':'text',},
            c_info={'type':'text', 'conditioning':c, 'unconditional_conditioning':u, 
                    'unconditional_guidance_scale':scale},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)
        tx = self.net.vae_decode(x, which='text', temperature=self.text_temperature)
        tx = [remove_duplicate_word(txi) for txi in tx]
        tx_combined = '\n'.join(tx)
        return tx_combined

    def inference_dcg(self, imctx, fcs_lvl, textctx, textstrength, seed):
        n_samples = self.n_sample_image
        sampler = self.sampler
        h, w = self.output_dim
        device = self.net.device

        c_info_list = []

        if (textctx is not None) and (textctx != "") and (textstrength != 0):
            ut = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
            ct = self.net.ctx_encode([textctx], which='text').repeat(n_samples, 1, 1)
            scale = self.scale_imgto*(1-textstrength) + self.scale_textto*textstrength

            c_info_list.append({
                'type':'text', 
                'conditioning':ct, 
                'unconditional_conditioning':ut,
                'unconditional_guidance_scale':scale,
                'ratio': textstrength, })
        else:
            scale = self.scale_imgto
            textstrength = 0

        BICUBIC = PIL.Image.Resampling.BICUBIC
        cx = imctx.resize([w, h], resample=BICUBIC)
        cx = tvtrans.ToTensor()(cx)[None].to(device).to(self.dtype)
        ci = self.net.ctx_encode(cx, which='image')

        if self.disentanglement_noglobal:
            ci_glb = ci[:, 0:1]
            ci_loc = ci[:, 1: ]
            ci_loc = self.adjust_rank_f(ci_loc, fcs_lvl)
            ci = torch.cat([ci_glb, ci_loc], dim=1).repeat(n_samples, 1, 1)
        else:
            ci = self.adjust_rank_f(ci, fcs_lvl).repeat(n_samples, 1, 1)

        c_info_list.append({
            'type':'image', 
            'conditioning':ci, 
            'unconditional_conditioning':torch.zeros_like(ci),
            'unconditional_guidance_scale':scale,
            'ratio': (1-textstrength), })

        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample_multicontext(
            steps=self.ddim_steps,
            x_info={'type':'image',},
            c_info_list=c_info_list,
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)

        imout = self.net.vae_decode(x, which='image')
        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return imout

    def inference_tcg(self, *args):
        args_imag = list(args[0:10]) + [None, None, None, None, None]*2
        args_rest = args[10:]
        imin, imout = self.inference_mcg(*args_imag, *args_rest)
        return imin, imout

    def inference_mcg(self, *args):
        imctx = [args[0:5], args[5:10], args[10:15], args[15:20]]
        textctx, textstrength, seed = args[20:]

        n_samples = self.n_sample_image
        sampler = self.sampler
        h, w = self.output_dim
        device = self.net.device

        c_info_list = []

        if (textctx is not None) and (textctx != "") and (textstrength != 0):
            ut = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
            ct = self.net.ctx_encode([textctx], which='text').repeat(n_samples, 1, 1)
            scale = self.scale_imgto*(1-textstrength) + self.scale_textto*textstrength

            c_info_list.append({
                'type':'text', 
                'conditioning':ct, 
                'unconditional_conditioning':ut,
                'unconditional_guidance_scale':scale,
                'ratio': textstrength, })
        else:
            scale = self.scale_imgto
            textstrength = 0

        input_save = []
        imc = []
        for im, imm, strength, fcs_lvl, use_mask in imctx:
            if (im is None) and (imm is None):
                continue
            BILINEAR = PIL.Image.Resampling.BILINEAR
            BICUBIC = PIL.Image.Resampling.BICUBIC
            if use_mask:
                cx = imm['image'].resize([w, h], resample=BICUBIC)
                cx = tvtrans.ToTensor()(cx)[None].to(self.dtype).to(device)
                m = imm['mask'].resize([w, h], resample=BILINEAR)
                m = tvtrans.ToTensor()(m)[None, 0:1].to(self.dtype).to(device)
                m = (1-m)
                cx_show = cx*m
                ci = self.net.ctx_encode(cx, which='image', masks=m)
            else:
                cx = im.resize([w, h], resample=BICUBIC)
                cx = tvtrans.ToTensor()(cx)[None].to(self.dtype).to(device)
                ci = self.net.ctx_encode(cx, which='image')
                cx_show = cx

            input_save.append(tvtrans.ToPILImage()(cx_show[0]))

            if self.disentanglement_noglobal:
                ci_glb = ci[:, 0:1]
                ci_loc = ci[:, 1: ]
                ci_loc = self.adjust_rank_f(ci_loc, fcs_lvl)
                ci = torch.cat([ci_glb, ci_loc], dim=1).repeat(n_samples, 1, 1)
            else:
                ci = self.adjust_rank_f(ci, fcs_lvl).repeat(n_samples, 1, 1)
            imc.append(ci * strength)

        cis = torch.cat(imc, dim=1)
        c_info_list.append({
            'type':'image', 
            'conditioning':cis, 
            'unconditional_conditioning':torch.zeros_like(cis),
            'unconditional_guidance_scale':scale,
            'ratio': (1-textstrength), })

        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample_multicontext(
            steps=self.ddim_steps,
            x_info={'type':'image',},
            c_info_list=c_info_list,
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)

        imout = self.net.vae_decode(x, which='image')
        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return input_save, imout

# vd_inference = vd_dummy()
vd_inference = vd_inference(which='v1.0', fp16=True)

#################
# sub interface #
#################

def t2i_interface(with_example=False):
    gr.HTML('<p id=myinst>&nbsp Description: ' + get_instruction("Text-to-Image") + '</p>')
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(lines=4, placeholder="Input prompt...", label='Text Input')
            seed = gr.Number(20, label="Seed", precision=0)
            button = gr.Button("Run")
        with gr.Column():
            img_output = gr.Gallery(label="Image Result", elem_id='customized_imbox').style(grid=n_sample_image)

    button.click(
        vd_inference.inference_t2i,
        inputs=[text, seed],
        outputs=[img_output])

    if with_example:
        gr.Examples(
            label='Examples',
            examples=get_example('Text-to-Image'),
            fn=vd_inference.inference_t2i,
            inputs=[text, seed],
            outputs=[img_output],
            cache_examples=cache_examples),

def i2i_interface(with_example=False):
    gr.HTML('<p id=myinst>&nbsp Description: ' + get_instruction("Image-Variation") + '</p>')
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label='Image Input', type='pil', elem_id='customized_imbox')
            sim_flag  = gr.Checkbox(label='Show Detail Controls')
            with gr.Row():
                fid_lvl = gr.Slider(label="Fidelity (Dislike -- Same)", minimum=0, maximum=1, value=0, step=0.02, visible=False)
                fcs_lvl = gr.Slider(label="Focus (Semantic -- Style)", minimum=0, maximum=1, value=0.5, step=0.02, visible=False)
            clr_adj = gr.Radio(label="Color Adjustment", choices=["None", "Simple"], value='Simple', visible=False)
            explain = gr.HTML('<p id=myinst>&nbsp Fidelity: How likely the output image looks like the referece image (0-dislike (default), 1-same).</p>'+
                              '<p id=myinst>&nbsp Focus: What the output image should focused on (0-semantic, 0.5-balanced (default), 1-style).</p>', 
                              visible=False)
            seed = gr.Number(20, label="Seed", precision=0)
            button = gr.Button("Run")
        with gr.Column():
            img_output = gr.Gallery(label="Image Result", elem_id='customized_imbox').style(grid=n_sample_image)

        sim_flag.change(
            fn=lambda x: {
                explain : gr.update(visible=x), 
                fid_lvl : gr.update(visible=x), 
                fcs_lvl : gr.update(visible=x), 
                clr_adj : gr.update(visible=x), },
            inputs=sim_flag,
            outputs=[explain, fid_lvl, fcs_lvl, clr_adj, seed],)

    button.click(
        vd_inference.inference_i2i,
        inputs=[img_input, fid_lvl, fcs_lvl, clr_adj, seed],
        outputs=[img_output])

    if with_example:
        gr.Examples(
            label='Examples',
            examples=get_example('Image-Variation'),
            fn=vd_inference.inference_i2i,
            inputs=[img_input, fid_lvl, fcs_lvl, clr_adj, seed],
            outputs=[img_output],
            cache_examples=cache_examples),

def i2t_interface(with_example=False):
    gr.HTML('<p id=myinst>&nbsp Description: ' + get_instruction("Image-to-Text") + '</p>')
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label='Image Input', type='pil', elem_id='customized_imbox')
            seed = gr.Number(20, label="Seed", precision=0)
            button = gr.Button("Run")
        with gr.Column():
            txt_output = gr.Textbox(lines=4, label='Text Result')

    button.click(
        vd_inference.inference_i2t,
        inputs=[img_input, seed],
        outputs=[txt_output])

    if with_example:
        gr.Examples(
            label='Examples',
            examples=get_example('Image-to-Text'),
            fn=vd_inference.inference_i2t,
            inputs=[img_input, seed],
            outputs=[txt_output],
            cache_examples=cache_examples),

def t2t_interface(with_example=False):
    gr.HTML('<p id=myinst>&nbsp Description: ' + get_instruction("Text-Variation") + '</p>')
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(lines=4, placeholder="Input prompt...", label='Text Input')
            seed = gr.Number(20, label="Seed", precision=0)
            button = gr.Button("Run")
        with gr.Column():
            txt_output = gr.Textbox(lines=4, label='Text Result')

    button.click(
        vd_inference.inference_t2t,
        inputs=[text, seed],
        outputs=[txt_output])

    if with_example:
        gr.Examples(
            label='Examples',
            examples=get_example('Text-Variation'),
            fn=vd_inference.inference_t2t,
            inputs=[text, seed],
            outputs=[txt_output],
            cache_examples=cache_examples, )

class image_mimage_swap(object):
    def __init__(self, block0, block1):
        self.block0 = block0
        self.block1 = block1
        self.which_update = 'both'

    def __call__(self, x0, x1, flag):
        if self.which_update == 'both':
            return self.update_both(x0, x1, flag)
        elif self.which_update == 'visible':
            return self.update_visible(x0, x1, flag)
        elif self.which_update == 'visible_oneoff':
            return self.update_visible_oneoff(x0, x1, flag)
        else:
            assert False

    def update_both(self, x0, x1, flag):
        if flag:
            ug0 = gr.update(visible=False)
            if x0 is None:
                ug1 = gr.update(value=None, visible=True)
            else:
                if (x1 is not None) and ('mask' in x1):
                    value1 = {'image':x0, 'mask':x1['mask']}
                else:
                    value1 = {'image':x0, 'mask':None}
                ug1 = gr.update(value=value1, visible=True)
        else:
            if (x1 is not None) and ('image' in x1):
                value0 = x1['image']
            else:
                value0 = None
            ug0 = gr.update(value=value0, visible=True)
            ug1 = gr.update(visible=False)
        return {
            self.block0 : ug0,
            self.block1 : ug1,}

    def update_visible(self, x0, x1, flag):
        return {
            self.block0 : gr.update(visible=not flag),
            self.block1 : gr.update(visible=flag), }

    def update_visible_oneoff(self, x0, x1, flag):
        self.which_update = 'both'
        return {
            self.block0 : gr.update(visible=not flag),
            self.block1 : gr.update(visible=flag), }

class example_visible_only_hack(object):
    def __init__(self, checkbox_list, functor_list):
        self.checkbox_list = checkbox_list
        self.functor_list = functor_list

    def __call__(self, *args):
        for bi, fi, vi in zip(self.checkbox_list, self.functor_list, args):
            if bi.value != vi:
                fi.which_update = 'visible_oneoff'

def dcg_interface(with_example=False):
    gr.HTML('<p id=myinst>&nbsp Description: ' + get_instruction("Dual-Context") + '</p>')
    with gr.Row():
        input_session = []
        with gr.Column():
            img = gr.Image(label='Image Input', type='pil', elem_id='customized_imbox')
            fcs = gr.Slider(label="Focus (Semantic -- Style)", minimum=0, maximum=1, value=0.5, step=0.02)
            gr.HTML('<p id=myinst>&nbsp Focus: Focus on what aspect of the image? (0-semantic, 0.5-balanced (default), 1-style).</p>')

            text = gr.Textbox(lines=2, placeholder="Input prompt...", label='Text Input')
            tstrength = gr.Slider(label="Text Domination (NoEffect -- TextOnly)", minimum=0, maximum=1, value=0, step=0.02)

            seed = gr.Number(20, label="Seed", precision=0)
            button = gr.Button("Run")

        with gr.Column():
            output_gallary = gr.Gallery(label="Image Result", elem_id='customized_imbox').style(grid=n_sample_image)

    input_list = []
    for i in input_session:
        input_list += i
    button.click(
        vd_inference.inference_dcg, 
        inputs=[img, fcs, text, tstrength, seed],
        outputs=[output_gallary])

    if with_example:
        gr.Examples(
            label='Examples',
            examples=get_example('Dual-Context'),
            fn=vd_inference.inference_dcg,
            inputs=[img, fcs, text, tstrength, seed],
            outputs=[output_gallary],
            cache_examples=cache_examples)

def tcg_interface(with_example=False):
    gr.HTML('<p id=myinst>&nbsp Description: ' + get_instruction("Triple-Context") + '</p>')
    with gr.Row():
        input_session = []
        with gr.Column(min_width=940):
            with gr.Row():
                with gr.Column():
                    img0  = gr.Image(label='Image Input', type='pil', elem_id='customized_imbox')
                    img0.as_example = types.MethodType(customized_as_example, img0)
                    imgm0 = gr.Image(label='Image Input with Mask', type='pil', elem_id='customized_imbox', tool='sketch', source="upload", visible=False)
                    imgm0.postprocess = types.MethodType(customized_postprocess, imgm0)
                    imgm0.as_example = types.MethodType(customized_as_example, imgm0)
                    istrength0 = gr.Slider(label="Weight", minimum=0, maximum=1, value=1, step=0.02)
                    fcs0 = gr.Slider(label="Focus (Semantic -- Style)", minimum=0, maximum=1, value=0.5, step=0.02)
                    msk0 = gr.Checkbox(label='Use mask?')
                    swapf0 = image_mimage_swap(img0, imgm0)

                    msk0.change(
                        fn=swapf0,
                        inputs=[img0, imgm0, msk0],
                        outputs=[img0, imgm0],)
                    input_session.append([img0, imgm0, istrength0, fcs0, msk0])

                with gr.Column():
                    img1  = gr.Image(label='Image Input', type='pil', elem_id='customized_imbox')
                    img1.as_example = types.MethodType(customized_as_example, img1)
                    imgm1 = gr.Image(label='Image Input with Mask', type='pil', elem_id='customized_imbox', tool='sketch', source="upload", visible=False)
                    imgm1.postprocess = types.MethodType(customized_postprocess, imgm1)
                    imgm1.as_example = types.MethodType(customized_as_example, imgm1)
                    istrength1 = gr.Slider(label="Weight", minimum=0, maximum=1, value=1, step=0.02)
                    fcs1 = gr.Slider(label="Focus (Semantic -- Style)", minimum=0, maximum=1, value=0.5, step=0.02)
                    msk1 = gr.Checkbox(label='Use mask?')
                    swapf1 = image_mimage_swap(img1, imgm1)

                    msk1.change(
                        fn=swapf1,
                        inputs=[img1, imgm1, msk1],
                        outputs=[img1, imgm1],)
                    input_session.append([img1, imgm1, istrength1, fcs1, msk1])

            gr.HTML('<p id=myinst>&nbsp Weight: The strength of the reference image. This weight is subject to <u>Text Domination</u>).</p>'+
                    '<p id=myinst>&nbsp Focus: Focus on what aspect of the image? (0-semantic, 0.5-balanced (default), 1-style).</p>'+
                    '<p id=myinst>&nbsp Mask: Remove regions on reference image so they will not influence the output.</p>',)

            text = gr.Textbox(lines=2, placeholder="Input prompt...", label='Text Input')
            tstrength = gr.Slider(label="Text Domination (NoEffect -- TextOnly)", minimum=0, maximum=1, value=0, step=0.02)

            seed = gr.Number(20, label="Seed", precision=0)
            button = gr.Button("Run")

        with gr.Column(min_width=470):
            input_gallary = gr.Gallery(label="Input Display", elem_id="customized_imbox").style(grid=2)
            output_gallary = gr.Gallery(label="Image Result", elem_id="customized_imbox").style(grid=n_sample_image)

    input_list = []
    for i in input_session:
        input_list += i
    input_list += [text, tstrength, seed]
    button.click(
        vd_inference.inference_tcg, 
        inputs=input_list,
        outputs=[input_gallary, output_gallary])

    if with_example:
        create_myexamples(
            label='Examples',
            examples=get_example('Triple-Context'),
            fn=vd_inference.inference_tcg,
            inputs=input_list,
            outputs=[input_gallary, output_gallary, ],
            cache_examples=cache_examples, )

    gr.HTML('<br><p id=myinst>&nbsp How to add mask: Please see the following instructions.</p><br>'+
            '<div id="maskinst">'+
                '<img src="file/assets/demo/misc/mask_inst1.gif">'+
                '<img src="file/assets/demo/misc/mask_inst2.gif">'+
                '<img src="file/assets/demo/misc/mask_inst3.gif">'+
            '</div>')

def mcg_interface(with_example=False):
    num_img_input = 4
    gr.HTML('<p id=myinst>&nbsp Description: ' + get_instruction("Multi-Context") + '</p>')
    with gr.Row():
        input_session = []
        with gr.Column():
            for idx in range(num_img_input):
                with gr.Tab('Image{}'.format(idx+1)):
                    img = gr.Image(label='Image Input', type='pil', elem_id='customized_imbox')
                    img.as_example = types.MethodType(customized_as_example, img)
                    imgm = gr.Image(label='Image Input with Mask', type='pil', elem_id='customized_imbox', tool='sketch', source="upload", visible=False)
                    imgm.postprocess = types.MethodType(customized_postprocess, imgm)
                    imgm.as_example = types.MethodType(customized_as_example, imgm)

                    with gr.Row():
                        istrength = gr.Slider(label="Weight", minimum=0, maximum=1, value=1, step=0.02)
                        fcs = gr.Slider(label="Focus (Semantic -- Style)", minimum=0, maximum=1, value=0.5, step=0.02)
                    msk = gr.Checkbox(label='Use mask?')
                    gr.HTML('<p id=myinst>&nbsp Weight: The strength of the reference image. This weight is subject to <u>Text Domination</u>).</p>'+
                            '<p id=myinst>&nbsp Focus: Focus on what aspect of the image? (0-semantic, 0.5-balanced (default), 1-style).</p>'+
                            '<p id=myinst>&nbsp Mask: Remove regions on reference image so they will not influence the output.</p>',)

                    msk.change(
                        fn=image_mimage_swap(img, imgm),
                        inputs=[img, imgm, msk],
                        outputs=[img, imgm],)
                    input_session.append([img, imgm, istrength, fcs, msk])

            text = gr.Textbox(lines=2, placeholder="Input prompt...", label='Text Input')
            tstrength = gr.Slider(label="Text Domination (NoEffect -- TextOnly)", minimum=0, maximum=1, value=0, step=0.02)

            seed = gr.Number(20, label="Seed", precision=0)
            button = gr.Button("Run")


        with gr.Column():
            input_gallary = gr.Gallery(label="Input Display", elem_id='customized_imbox').style(grid=4)
            output_gallary = gr.Gallery(label="Image Result", elem_id='customized_imbox').style(grid=n_sample_image)

    input_list = []
    for i in input_session:
        input_list += i
    input_list += [text, tstrength, seed]
    button.click(
        vd_inference.inference_mcg, 
        inputs=input_list,
        outputs=[input_gallary, output_gallary], )

    if with_example:
        create_myexamples(
            label='Examples',
            examples=get_example('Multi-Context'),
            fn=vd_inference.inference_mcg,
            inputs=input_list,
            outputs=[input_gallary, output_gallary],
            cache_examples=cache_examples, )

    gr.HTML('<br><p id=myinst>&nbsp How to add mask: Please see the following instructions.</p><br>'+
            '<div id="maskinst">'+
                '<img src="file/assets/demo/misc/mask_inst1.gif">'+
                '<img src="file/assets/demo/misc/mask_inst2.gif">'+
                '<img src="file/assets/demo/misc/mask_inst3.gif">'+
            '</div>')

###########
# Example #
###########

def get_example(mode):
    if mode == 'Text-to-Image':
        case = [
            ['a dream of a village in china, by Caspar David Friedrich, matte painting trending on artstation HQ', 23],
            ['a beautiful landscape with mountains and rivers', 20],
        ]
    elif mode == "Image-Variation":
        case = [
            ['assets/demo/reg_example/ghibli.jpg', 0, 0.5, 'None', 20],
            ['assets/demo/reg_example/ghibli.jpg', 0.5, 0.5, 'None', 20],
            ['assets/demo/reg_example/matisse.jpg', 0, 0, 'None', 20],
            ['assets/demo/reg_example/matisse.jpg', 0, 1, 'Simple', 20],
            ['assets/demo/reg_example/vermeer.jpg', 0.2, 0.3, 'None', 30],
        ]
    elif mode == "Image-to-Text":
        case = [
            ['assets/demo/reg_example/house_by_lake.jpg', 20],
        ]
    elif mode == "Text-Variation":
        case = [
            ['heavy arms gundam penguin mech', 20],
        ]
    elif mode == "Dual-Context":
        case = [
            ['assets/demo/reg_example/benz.jpg', 0.5, 'cyberpunk 2077', 0.7, 22],
            ['assets/demo/reg_example/ghibli.jpg', 1, 'Red maple on a hill in golden Autumn.', 0.5, 21],
        ]
    elif mode == "Triple-Context":
        case = [
            [
                'assets/demo/reg_example/night_light.jpg', None, 1   , 0.5, False,
                'assets/demo/reg_example/paris.jpg'      , None, 0.94, 0.5, False,
                "snow on the street", 0.4, 28],
            [
                'assets/demo/tcg_example/e1i0.jpg', None, 1  , 0.5, False,
                'assets/demo/tcg_example/e1i1.jpg', None, 0.94, 0.5, False,
                "a painting of an elegant woman in front of the moon", 0.2, 217],
            [
                'assets/demo/tcg_example/e2i0.jpg',  None, 1, 0.5, False,
                'assets/demo/reg_example/paris.jpg', None, 1, 0.5, False,
                "", 0, 29],
            [
                'assets/demo/tcg_example/e0i0.jpg', None, 1  , 0.5, False,
                'assets/demo/tcg_example/e0i1.jpg', None, 0.9, 0.5, False,
                "rose blooms on the tree", 0.2, 20],
            [
                'assets/demo/reg_example/ghibli.jpg', None, 1   , 1  , False,
                'assets/demo/reg_example/space.jpg' , None, 0.88, 0.5, False,
                "", 0, 20],
            [
                'assets/demo/reg_example/train.jpg'  , None, 0.8, 0.5, False,
                'assets/demo/reg_example/matisse.jpg', None, 1  , 1  , False,
                "", 0, 20],
        ]
    elif mode == "Multi-Context":
        case = [
            [
                'assets/demo/mcg_example/e0i0.jpg', None, 1, 0.5, False,
                'assets/demo/mcg_example/e0i1.jpg', None, 1, 0.5, False,
                'assets/demo/mcg_example/e0i2.jpg', None, 0.86, 0.5, False,
                None, None, 1, 0.5, False,
                "", 0, 20],
        ]
    else:
        raise ValueError
    return case

#############
# Interface #
#############

css = """
    #customized_imbox {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"] {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"]>div {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"]>iframe {
        min-height: 450px;
    }
    #customized_imbox>div.unpadded_box {
        min-height: 450px;
    }
    #myinst {
        font-size: 0.8rem; 
        margin: 0rem;
        color: #6B7280;
    }
    #maskinst {
        text-align: justify;
        min-width: 1200px;
    }
    #maskinst>img {
        min-width:399px;
        max-width:450px;
        vertical-align: top;
        display: inline-block;
    }
    #maskinst:after {
        content: "";
        width: 100%;
        display: inline-block;
    }
"""

if True:
    with gr.Blocks(css=css) as demo:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
            <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                Versatile Diffusion
            </h1>
            <h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
            We built <b>Versatile Diffusion (VD), the first unified multi-flow multimodal diffusion framework</b>, as a step towards <b>Universal Generative AI</b>. 
            VD can natively support image-to-text, image-variation, text-to-image, and text-variation, 
            and can be further extended to other applications such as 
            semantic-style disentanglement, image-text dual-guided generation, latent image-to-text-to-image editing, and more. 
            Future versions will support more modalities such as speech, music, video and 3D. 
            </h2>
            <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem">
            Xingqian Xu, Atlas Wang, Eric Zhang, Kai Wang, 
            and <a href="https://www.humphreyshi.com/home">Humphrey Shi</a> 
            [<a href="https://arxiv.org/abs/2211.08332" style="color:blue;">arXiv</a>] 
            [<a href="https://github.com/SHI-Labs/Versatile-Diffusion" style="color:blue;">GitHub</a>]
            </h3>
            </div>
            """)

        with gr.Tab('Text-to-Image'):
            t2i_interface(with_example=True)
        with gr.Tab('Image-Variation'):
            i2i_interface(with_example=True)
        with gr.Tab('Image-to-Text'):
            i2t_interface(with_example=True)
        with gr.Tab('Text-Variation'):
            t2t_interface(with_example=True)
        with gr.Tab('Dual-Context Image-Generation'):
            dcg_interface(with_example=True)
        with gr.Tab('Triple-Context Image-Blender'):
            tcg_interface(with_example=True)
        with gr.Tab('Multi-Context Image-Blender'):
            mcg_interface(with_example=True)

        gr.HTML(
            """
            <div style="text-align: justify; max-width: 1200px; margin: 20px auto;">
            <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
            <b>Version</b>: {}
            </h3>
            <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
            <b>Caution</b>: 
            We would like the raise the awareness of users of this demo of its potential issues and concerns.
            Like previous large foundation models, Versatile Diffusion could be problematic in some cases, partially due to the imperfect training data and pretrained network (VAEs / context encoders) with limited scope.
            In its future research phase, VD may do better on tasks such as text-to-image, image-to-text, etc., with the help of more powerful VAEs, more sophisticated network designs, and more cleaned data.
            So far, we keep all features available for research testing both to show the great potential of the VD framework and to collect important feedback to improve the model in the future.
            We welcome researchers and users to report issues with the HuggingFace community discussion feature or email the authors.
            </h3>
            <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
            <b>Biases and content acknowledgement</b>:
            Beware that VD may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography, and violence. 
            VD was trained on the LAION-2B dataset, which scraped non-curated online images and text, and may contained unintended exceptions as we removed illegal content. 
            VD in this demo is meant only for research purposes.
            </h3>
            </div>
            """.format(' '+vd_inference.which))

    demo.launch(share=True)
    # demo.launch(debug=True)
