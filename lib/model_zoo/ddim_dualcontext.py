import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from .diffusion_utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

from .ddim import DDIMSampler

class DDIMSampler_DualContext(DDIMSampler):
    @torch.no_grad()
    def sample_text(self, *args, **kwargs):
        self.cond_type = 'prompt'
        return self.sample(*args, **kwargs)

    @torch.no_grad()
    def sample_vision(self, *args, **kwargs):
        self.cond_type = 'vision'
        return self.sample(*args, **kwargs)

    @torch.no_grad()
    def sample_mixed(self, *args, **kwargs):
        self.cond_type = kwargs.pop('cond_mixed_p')
        return self.sample(*args, **kwargs)

    @torch.no_grad()
    def sample(self,
               steps,
               shape,
               xt=None,
               conditioning=None,
               eta=0.,
               temperature=1.,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        # sampling
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')

        samples, intermediates = self.ddim_sampling(
            conditioning, 
            shape,
            xt=xt,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, 
                      conditioning,
                      shape,
                      xt=None, 
                      ddim_use_original_steps=False,
                      timesteps=None, 
                      log_every_t=100,
                      temperature=1., 
                      noise_dropout=0., 
                      unconditional_guidance_scale=1., 
                      unconditional_conditioning=None,):
        device = self.model.betas.device
        bs = shape[0]
        if xt is None:
            img = torch.randn(shape, device=device)
        else:
            img = xt

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(img, conditioning, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      temperature=temperature,
                                      noise_dropout=noise_dropout,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, conditioning, t, index, repeat_noise=False, use_original_steps=False, 
                      temperature=1., noise_dropout=0.,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, conditioning, cond_type=self.cond_type)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            # c_in = torch.cat([unconditional_conditioning, conditioning])

            # Added for vd-dc dual guidance
            if isinstance(unconditional_conditioning, list):
                c_in = [torch.cat([ui, ci]) for ui, ci in zip(unconditional_conditioning, conditioning)]
            else:
                c_in = torch.cat([unconditional_conditioning, conditioning])

            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, cond_type=self.cond_type).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
