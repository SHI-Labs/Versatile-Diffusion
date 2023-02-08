"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from .diffusion_utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, 
                                                  num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,
                                                  verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,verbose=verbose)

        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               steps,
               shape,
               x_info,
               c_info,
               eta=0.,
               temperature=1.,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self.ddim_sampling(
            shape,
            x_info=x_info,
            c_info=c_info,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, 
                      shape,
                      x_info,
                      c_info,
                      noise_dropout=0., 
                      temperature=1., 
                      log_every_t=100,):

        device = self.model.device
        dtype = c_info['conditioning'].dtype
        bs = shape[0]
        timesteps = self.ddim_timesteps
        if ('xt' in x_info) and (x_info['xt'] is not None):
            xt = x_info['xt'].astype(dtype).to(device)
            x_info['x'] = xt
        elif ('x0' in x_info) and (x_info['x0'] is not None):
            x0 = x_info['x0'].type(dtype).to(device)
            ts = timesteps[x_info['x0_forward_timesteps']].repeat(bs)
            ts = torch.Tensor(ts).long().to(device)
            timesteps = timesteps[:x_info['x0_forward_timesteps']]
            x0_nz = self.model.q_sample(x0, ts)
            x_info['x'] = x0_nz
        else:
            x_info['x'] = torch.randn(shape, device=device, dtype=dtype)
            
        intermediates = {'pred_xt': [], 'pred_x0': []}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(
                x_info, c_info, ts, index, 
                noise_dropout=noise_dropout,
                temperature=temperature,)
            pred_xt, pred_x0 = outs
            x_info['x'] = pred_xt

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x_info, c_info, t, index, 
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,):

        x = x_info['x']
        unconditional_guidance_scale = c_info['unconditional_guidance_scale']

        b, *_, device = *x.shape, x.device
        if unconditional_guidance_scale == 1.:
            c_info['c'] = c_info['conditioning']
            e_t = self.model.apply_model(x_info, t, c_info)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([c_info['unconditional_conditioning'], c_info['conditioning']])
            x_info['x'] = x_in
            c_info['c'] = c_in
            e_t_uncond, e_t = self.model.apply_model(x_info, t_in, c_info).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        extended_shape = [b] + [1]*(len(e_t.shape)-1) 
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
    def sample_multicontext(self,
                            steps,
                            shape,
                            x_info,
                            c_info_list,
                            eta=0.,
                            temperature=1.,
                            noise_dropout=0.,
                            verbose=True,
                            log_every_t=100,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self.ddim_sampling_multicontext(
            shape,
            x_info=x_info,
            c_info_list=c_info_list,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_multicontext(self, 
                                   shape,
                                   x_info,
                                   c_info_list,
                                   noise_dropout=0., 
                                   temperature=1., 
                                   log_every_t=100,):

        device = self.model.device
        dtype = c_info_list[0]['conditioning'].dtype
        bs = shape[0]
        timesteps = self.ddim_timesteps
        if ('xt' in x_info) and (x_info['xt'] is not None):
            xt = x_info['xt'].astype(dtype).to(device)
            x_info['x'] = xt
        elif ('x0' in x_info) and (x_info['x0'] is not None):
            x0 = x_info['x0'].type(dtype).to(device)
            ts = timesteps[x_info['x0_forward_timesteps']].repeat(bs)
            ts = torch.Tensor(ts).long().to(device)
            timesteps = timesteps[:x_info['x0_forward_timesteps']]
            x0_nz = self.model.q_sample(x0, ts)
            x_info['x'] = x0_nz
        else:
            x_info['x'] = torch.randn(shape, device=device, dtype=dtype)
            
        intermediates = {'pred_xt': [], 'pred_x0': []}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim_multicontext(
                x_info, c_info_list, ts, index, 
                noise_dropout=noise_dropout,
                temperature=temperature,)
            pred_xt, pred_x0 = outs
            x_info['x'] = pred_xt

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample_ddim_multicontext(
            self, x_info, c_info_list, t, index, 
            repeat_noise=False, 
            use_original_steps=False, 
            noise_dropout=0.,
            temperature=1.,):

        x = x_info['x']
        b, *_, device = *x.shape, x.device
        unconditional_guidance_scale = None

        for c_info in c_info_list:
            if unconditional_guidance_scale is None:
                unconditional_guidance_scale = c_info['unconditional_guidance_scale']
            else:
                assert unconditional_guidance_scale==c_info['unconditional_guidance_scale'], \
                    "A different unconditional guidance scale between different context is not allowed!"

            if unconditional_guidance_scale == 1.:
                c_info['c'] = c_info['conditioning']
                
            else:
                c_in = torch.cat([c_info['unconditional_conditioning'], c_info['conditioning']])
                c_info['c'] = c_in

        if unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model_multicontext(x_info, t, c_info_list)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            x_info['x'] = x_in
            e_t_uncond, e_t = self.model.apply_model_multicontext(x_info, t_in, c_info_list).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        extended_shape = [b] + [1]*(len(e_t.shape)-1) 
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
