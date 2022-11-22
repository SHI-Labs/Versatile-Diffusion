import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from .diffusion_utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

from .ddim import DDIMSampler

class DDIMSampler_VD(DDIMSampler):
    @torch.no_grad()
    def sample(self,
               steps,
               shape,
               xt=None,
               conditioning=None,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               xtype='image',
               ctype='prompt',
               eta=0.,
               temperature=1.,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self.ddim_sampling(
            shape,
            xt=xt,
            conditioning=conditioning, 
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            xtype=xtype,
            ctype=ctype,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, 
                      shape,
                      xt=None,
                      conditioning=None,
                      unconditional_guidance_scale=1., 
                      unconditional_conditioning=None,
                      xtype='image',
                      ctype='prompt',
                      ddim_use_original_steps=False,
                      timesteps=None, 
                      noise_dropout=0., 
                      temperature=1., 
                      log_every_t=100,):

        device = self.model.device
        bs = shape[0]
        if xt is None:
            xt = torch.randn(shape, device=device, dtype=conditioning.dtype)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'pred_xt': [], 'pred_x0': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        pred_xt = xt
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(
                pred_xt, conditioning, ts, index, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning, 
                xtype=xtype,
                ctype=ctype,
                use_original_steps=ddim_use_original_steps,
                noise_dropout=noise_dropout,
                temperature=temperature,)
            pred_xt, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, conditioning, t, index, 
                      unconditional_guidance_scale=1., 
                      unconditional_conditioning=None, 
                      xtype='image',
                      ctype='prompt',
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,):

        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, conditioning, xtype=xtype, ctype=ctype)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, conditioning])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, xtype=xtype, ctype=ctype).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)

        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample_dc(self,
               steps,
               shape,
               xt=None,
               first_conditioning=None,
               second_conditioning=None,
               unconditional_guidance_scale=1.,
               xtype='image',
               first_ctype='prompt',
               second_ctype='prompt',
               eta=0.,
               temperature=1.,
               mixed_ratio=0.5,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self.ddim_sampling_dc(
            shape,
            xt=xt,
            first_conditioning=first_conditioning,
            second_conditioning=second_conditioning,
            unconditional_guidance_scale=unconditional_guidance_scale,
            xtype=xtype,
            first_ctype=first_ctype,
            second_ctype=second_ctype,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,
            mixed_ratio=mixed_ratio, )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_dc(self, 
                      shape,
                      xt=None,
                      first_conditioning=None,
                      second_conditioning=None,
                      unconditional_guidance_scale=1., 
                      xtype='image',
                      first_ctype='prompt',
                      second_ctype='prompt',
                      ddim_use_original_steps=False,
                      timesteps=None, 
                      noise_dropout=0., 
                      temperature=1.,
                      mixed_ratio=0.5,
                      log_every_t=100,):

        device = self.model.device
        bs = shape[0]
        if xt is None:
            xt = torch.randn(shape, device=device, dtype=first_conditioning[1].dtype)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'pred_xt': [], 'pred_x0': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        pred_xt = xt
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim_dc(
                pred_xt, 
                first_conditioning, 
                second_conditioning, 
                ts, index, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                xtype=xtype,
                first_ctype=first_ctype,
                second_ctype=second_ctype,
                use_original_steps=ddim_use_original_steps,
                noise_dropout=noise_dropout,
                temperature=temperature,
                mixed_ratio=mixed_ratio,)
            pred_xt, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample_ddim_dc(self, x, 
                      first_conditioning,
                      second_conditioning,
                      t, index, 
                      unconditional_guidance_scale=1., 
                      xtype='image',
                      first_ctype='prompt',
                      second_ctype='prompt',
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,
                      mixed_ratio=0.5,):

        b, *_, device = *x.shape, x.device

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        first_c = torch.cat(first_conditioning)
        second_c = torch.cat(second_conditioning)

        e_t_uncond, e_t = self.model.apply_model_dc(
            x_in, t_in, first_c, second_c, xtype=xtype, first_ctype=first_ctype, second_ctype=second_ctype, mixed_ratio=mixed_ratio).chunk(2)

        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if xtype == 'image':
            extended_shape = (b, 1, 1, 1)
        elif xtype == 'text':
            extended_shape = (b, 1)

        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
