import gradio as gr
import os
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr
from contextlib import nullcontext

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD, DDIMSampler_VD_DualContext
from lib.model_zoo.ddim_dualcontext import DDIMSampler_DualContext

from lib.experiments.sd_default import color_adjust

class vd_inference(object):
    def __init__(self, type='official'):
        if type in ['dc', '2-flow']:
            cfgm_name = 'vd_dc_noema'
            sampler = DDIMSampler_DualContext
            pth = 'pretrained/vd-dc.pth'
        elif type in ['official', '4-flow']:
            cfgm_name = 'vd_noema'
            sampler = DDIMSampler_VD
            pth = 'pretrained/vd-official.pth'
        cfgm = model_cfg_bank()(cfgm_name)
        net = get_model()(cfgm)

        sd = torch.load(pth, map_location='cpu')
        net.load_state_dict(sd, strict=False)
        
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            net.to('cuda')
        self.model_name = cfgm_name
        self.net = net
        self.sampler = sampler(net)

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
        if self.use_cuda:
            x = x.to('cuda')
        return x

    def decode(self, z, xtype, ctype, color_adj='None', color_adj_to=None):
        net = self.net
        if xtype == 'image':
            x = net.autokl_decode(z)

            color_adj_flag = (color_adj!='None') and (color_adj is not None)
            color_adj_simple = color_adj=='Simple'
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
            n_samples = 2 if n_samples is None else n_samples
        elif xtype == 'text':
            n_samples = 4 if n_samples is None else n_samples

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

    def application_disensemble(self, cin, n_samples=2, level=0, color_adj=None,):
        net = self.net
        scale = 7.5
        sampler = self.sampler
        ddim_steps = 50
        ddim_eta = 0.0

        cin = self.regularize_image(cin)
        ctemp = cin*2 - 1
        ctemp = ctemp[None].repeat(n_samples, 1, 1, 1)
        c = net.clip_encode_vision(ctemp)
        u = None
        if scale != 1.0:
            dummy = torch.zeros_like(ctemp)
            u = net.clip_encode_vision(dummy)

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

        u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))        

        if demean:
            x_lowrank += x_mean
        return x_lowrank

    def remove_low_rank(self, x, demean=True, q=20, niter=10, q_remove=10):
        if demean:
            x_mean = x.mean(-1, keepdim=True)
            x_input = x - x_mean
        else:
            x_input = x

        u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
        s[:, 0:q_remove] = 0
        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))        

        if demean:
            x_lowrank += x_mean
        return x_lowrank

    def application_dualguided(self, cim, ctx, n_samples=2, mixing=0.5, color_adj=None, ):
        net = self.net
        scale = 7.5
        sampler = DDIMSampler_VD_DualContext(net)
        ddim_steps = 50
        ddim_eta = 0.0

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

    def application_i2t2i(self, cim, ctx_n, ctx_p, n_samples=2, color_adj=None,):
        net = self.net
        scale = 7.5
        sampler = DDIMSampler_VD_DualContext(net)
        ddim_steps = 50
        ddim_eta = 0.0
        prompt_temperature = 1.0

        ctemp0 = self.regularize_image(cim)
        ctemp1 = ctemp0*2 - 1
        ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1)
        cim = net.clip_encode_vision(ctemp1)
        uim = None
        if scale != 1.0:
            dummy = torch.zeros_like(ctemp1)
            uim = net.clip_encode_vision(dummy)

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

vd_inference = vd_inference('official')

def main(mode,
         image=None,
         prompt=None,
         nprompt=None,
         pprompt=None,
         color_adj=None,
         disentanglement_level=None,
         dual_guided_mixing=None,
         seed=0,):

    if seed<0:
        seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed+100)

    if mode == 'Text-to-Image':
        if (prompt is None) or (prompt == ""):
            return None, None
        with torch.no_grad():
            rv = vd_inference.inference(
                xtype = 'image',
                cin = prompt,
                ctype = 'prompt', )
        return rv, None
    elif mode == 'Image-Variation':
        if image is None:
            return None, None
        with torch.no_grad():
            rv = vd_inference.inference(
                xtype = 'image',
                cin = image,
                ctype = 'vision',
                color_adj = color_adj,)
        return rv, None
    elif mode == 'Image-to-Text':
        if image is None:
            return None, None
        with torch.no_grad():
            rv = vd_inference.inference(
                xtype = 'text',
                cin = image,
                ctype = 'vision',)
        return None, '\n'.join(rv)
    elif mode == 'Text-Variation':
        if prompt is None:
            return None, None
        with torch.no_grad():
            rv = vd_inference.inference(
                xtype = 'text',
                cin = prompt,
                ctype = 'prompt',)
        return None, '\n'.join(rv)
    elif mode == 'Disentanglement':
        if image is None:
            return None, None
        with torch.no_grad():
            rv = vd_inference.application_disensemble(
                cin = image,
                level = disentanglement_level,
                color_adj = color_adj,)
        return rv, None
    elif mode == 'Dual-Guided':
        if (image is None) or (prompt is None) or (prompt==""):
            return None, None
        with torch.no_grad():
            rv = vd_inference.application_dualguided(
                cim = image,
                ctx = prompt,
                mixing = dual_guided_mixing,
                color_adj = color_adj,)
        return rv, None
    elif mode == 'Latent-I2T2I':
        if (image is None) or (nprompt is None) or (nprompt=="") \
                or (pprompt is None) or (pprompt==""):
            return None, None
        with torch.no_grad():
            rv = vd_inference.application_i2t2i(
                cim = image,
                ctx_n = nprompt,
                ctx_p = pprompt,
                color_adj = color_adj,)
        return rv, None
    else:
        assert False, "No such mode!"

def get_instruction(mode):
    t2i_instruction = ["Generate image from text prompt."]
    i2i_instruction = [
        "Generate image conditioned on reference image.", 
        "Color Calibration provide an opinion to adjust image color according to reference image.", ]
    i2t_instruction = ["Generate text from reference image."]
    t2t_instruction = ["Generate text from reference text prompt. (Model insufficiently trained, thus results are still experimental)"]
    dis_instruction = [
        "Generate a variation of reference image that disentangled for semantic or style.",
        "Color Calibration provide an opinion to adjust image color according to reference image.",
        "Disentanglement level controls the level of focus towards semantic (-2, -1) or style (1 2). Level 0 serves as Image-Variation.", ]
    dug_instruction = [
        "Generate image from dual guidance of reference image and text prompt.",
        "Color Calibration provide an opinion to adjust image color according to reference image.",
        "Guidance Mixing provides linear balances between image and text context. (0 towards image, 1 towards text)", ]
    iti_instruction = [
        "Generate image variations via image-to-text, text-latent-editing, and then text-to-image. (Still under exploration)",
        "Color Calibration provide an opinion to adjust image color according to reference image.",
        "Input prompt that will be substract from text/text latent code.",
        "Input prompt that will be added to text/text latent code.", ]

    if mode == "Text-to-Image":
        return '\n'.join(t2i_instruction)
    elif mode == "Image-Variation":
        return '\n'.join(i2i_instruction)
    elif mode == "Image-to-Text":
        return '\n'.join(i2t_instruction)
    elif mode == "Text-Variation":
        return '\n'.join(t2t_instruction)
    elif mode == "Disentanglement":
        return '\n'.join(dis_instruction)
    elif mode == "Dual-Guided":
        return '\n'.join(dug_instruction)
    elif mode == "Latent-I2T2I":
        return '\n'.join(iti_instruction)

#############
# Interface #
#############

if True:
    img_output = gr.Gallery(label="Image Result").style(grid=2)
    txt_output = gr.Textbox(lines=4, label='Text Result', visible=False)  

    with gr.Blocks() as demo:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
            <h1 style="font-weight: 900; font-size: 3rem;">
                Versatile Diffusion
            </h1>
            <br>
            <h2 style="font-weight: 450; font-size: 1rem;">
            We built <b>Versatile Diffusion (VD), the first unified multi-flow multimodal diffusion framework</b>, as a step towards <b>Universal Generative AI</b>. 
            VD can natively handle image-to-text, image-variation, text-to-image, and text-variation, 
            and can be further extended to other applications such as 
            semantic-style disentanglement, image-text dual-guided generation, latent image-to-text-to-image editing, and more. 
            Future versions will support more modalities such as speech, music, video and 3D. 
            </h2>
            <br>
            <h3>Xingqian Xu, Atlas Wang, Eric Zhang, Kai Wang, 
            and <a href="https://www.humphreyshi.com/home">Humphrey Shi</a> 
            [<a href="url" style="color:blue;">arXiv</a>] 
            [<a href="https://github.com/SHI-Labs/Versatile-Diffusion" style="color:blue;">GitHub</a>]
            </h3>
            </div>
            """)
        mode_input = gr.Radio([
            "Text-to-Image", "Image-Variation", "Image-to-Text", "Text-Variation",
            "Disentanglement", "Dual-Guided", "Latent-I2T2I"], value='Text-to-Image', label="VD Flows and Applications")

        instruction = gr.Textbox(get_instruction("Text-to-Image"), label='Info')

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label='Image Input', visible=False)
                txt_input = gr.Textbox(lines=4, placeholder="Input prompt...", label='Text Input')
                ntxt_input = gr.Textbox(label='Remove Prompt', visible=False)
                ptxt_input = gr.Textbox(label='Add Prompt', visible=False)
                coladj_input = gr.Radio(["None", "Simple"], value='Simple', label="Color Calibration", visible=False)
                dislvl_input = gr.Slider(-2, 2, value=0, step=1, label="Disentanglement level", visible=False)
                dguide_input = gr.Slider(0, 1, value=0.5, step=0.01, label="Guidance Mixing", visible=False)
                seed_input = gr.Number(100, label="Seed", precision=0)

                btn = gr.Button("Run")
                btn.click(
                    main, 
                    inputs=[
                        mode_input,
                        img_input,
                        txt_input,
                        ntxt_input,
                        ptxt_input,
                        coladj_input,
                        dislvl_input,
                        dguide_input,
                        seed_input, ],
                    outputs=[img_output, txt_output])

            with gr.Column():
                img_output.render()
                txt_output.render()

        example_mode = [
            "Text-to-Image", 
            "Image-Variation", 
            "Image-to-Text", 
            "Text-Variation", 
            "Disentanglement", 
            "Dual-Guided", 
            "Latent-I2T2I"]

        def get_example(mode):
            if mode == 'Text-to-Image':
                case = [
                    ['a dream of a village in china, by Caspar David Friedrich, matte painting trending on artstation HQ', 23],
                    ['a beautiful grand nebula in the universe', 24],
                    ['heavy arms gundam penguin mech', 25],
                ]
            elif mode == "Image-Variation":
                case = [
                    ['assets/space.jpg', 'None', 26],
                    ['assets/train.jpg', 'Simple', 27],
                ]
            elif mode == "Image-to-Text":
                case = [
                    ['assets/boy_and_girl.jpg' , 28],
                    ['assets/house_by_lake.jpg', 29],
                ]
            elif mode == "Text-Variation":
                case = [
                    ['a dream of a village in china, by Caspar David Friedrich, matte painting trending on artstation HQ' , 32],
                    ['a beautiful grand nebula in the universe' , 33],
                    ['heavy arms gundam penguin mech', 34],
                ]
            elif mode == "Disentanglement":
                case = [
                    ['assets/vermeer.jpg', 'Simple', -2, 30],
                    ['assets/matisse.jpg', 'Simple',  2, 31],
                ]
            elif mode == "Dual-Guided":
                case = [
                    ['assets/benz.jpg',    'cyberpunk 2077', 'Simple', 0.75, 22],
                    ['assets/vermeer.jpg', 'a girl with a diamond necklace',  'Simple', 0.66, 21],
                ]
            elif mode == "Latent-I2T2I":
                case = [
                    ['assets/ghibli.jpg',  'white house', 'tall castle', 'Simple', 20],
                    ['assets/matisse.jpg', 'fruits and bottles on the table', 'flowers on the table', 'Simple', 21],
                ]
            else:
                raise ValueError
            case = [[mode] + casei for casei in case]
            return case

        def get_example_input(mode):
            if mode == 'Text-to-Image':
                inps = [txt_input, seed_input]
            elif mode == "Image-Variation":
                inps = [img_input, coladj_input, seed_input]
            elif mode == "Image-to-Text":
                inps = [img_input, seed_input]
            elif mode == "Text-Variation":
                inps = [txt_input, seed_input]
            elif mode == "Disentanglement":
                inps = [img_input, coladj_input, dislvl_input, seed_input]
            elif mode == "Dual-Guided":
                inps = [img_input, txt_input, coladj_input, dguide_input, seed_input]
            elif mode == "Latent-I2T2I":
                inps = [img_input, ntxt_input, ptxt_input, coladj_input, seed_input]
            else:
                raise ValueError
            return [mode_input] + inps

        with gr.Row():
            for emode in example_mode[0:4]:
                with gr.Column():
                    gr.Examples(
                        label=emode+' Examples',
                        examples=get_example(emode),
                        inputs=get_example_input(emode))
        with gr.Row():
            for emode in example_mode[4:7]:
                with gr.Column():
                    gr.Examples(
                        label=emode+' Examples',
                        examples=get_example(emode),
                        inputs=get_example_input(emode))

        mode_input.change(
            fn=lambda x: gr.update(value=get_instruction(x)),
            inputs=mode_input,
            outputs=instruction,)

        mode_input.change(
            fn=lambda x: gr.update(visible=(x not in ['Text-to-Image', 'Text-Variation'])),
            inputs=mode_input,
            outputs=img_input,)

        mode_input.change(
            fn=lambda x: gr.update(visible=(x in ['Text-to-Image', 'Text-Variation', 'Dual-Guided'])),
            inputs=mode_input,
            outputs=txt_input,)

        mode_input.change(
            fn=lambda x: gr.update(visible=(x in ['Latent-I2T2I'])),
            inputs=mode_input,
            outputs=ntxt_input,)
        mode_input.change(
            fn=lambda x: gr.update(visible=(x in ['Latent-I2T2I'])),
            inputs=mode_input,
            outputs=ptxt_input,)

        mode_input.change(
            fn=lambda x: gr.update(visible=(x not in ['Text-to-Image', 'Image-to-Text', 'Text-Variation'])),
            inputs=mode_input,
            outputs=coladj_input,)

        mode_input.change(
            fn=lambda x: gr.update(visible=(x=='Disentanglement')),
            inputs=mode_input,
            outputs=dislvl_input,)

        mode_input.change(
            fn=lambda x: gr.update(visible=(x=='Dual-Guided')),
            inputs=mode_input,
            outputs=dguide_input,)

        mode_input.change(
            fn=lambda x: gr.update(visible=(x not in ['Image-to-Text', 'Text-Variation'])),
            inputs=mode_input,
            outputs=img_output,)
        mode_input.change(
            fn=lambda x: gr.update(visible=(x in ['Image-to-Text', 'Text-Variation'])),
            inputs=mode_input,
            outputs=txt_output,)

    demo.launch(share=True)
