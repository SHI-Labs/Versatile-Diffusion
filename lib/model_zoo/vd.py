import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import copy
from functools import partial
from contextlib import contextmanager
from lib.model_zoo.common.get_model import get_model, register
from lib.log_service import print_log

symbol = 'vd'

from .diffusion_utils import \
    count_params, extract_into_tensor, make_beta_schedule
from .distributions import normal_kl, DiagonalGaussianDistribution

from .autokl import AutoencoderKL
from .ema import LitEma

def highlight_print(info):
    print_log('')
    print_log(''.join(['#']*(len(info)+4)))
    print_log('# '+info+' #')
    print_log(''.join(['#']*(len(info)+4)))
    print_log('')

class String_Reg_Buffer(nn.Module):
    def __init__(self, output_string):
        super().__init__()
        torch_string = torch.ByteTensor(list(bytes(output_string, 'utf8')))
        self.register_buffer('output_string', torch_string)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        list_str = self.output_string.tolist()
        output_string = bytes(list_str)
        output_string = output_string.decode()
        return output_string

@register('vd_v2_0')
class VD_v2_0(nn.Module):
    def __init__(self,
                 vae_cfg_list,
                 ctx_cfg_list,
                 diffuser_cfg_list,
                 global_layer_ptr=None,

                 parameterization="eps",
                 timesteps=1000,
                 use_ema=False,

                 beta_schedule="linear",
                 beta_linear_start=1e-4,
                 beta_linear_end=2e-2,
                 given_betas=None,
                 cosine_s=8e-3,

                 loss_type="l2",
                 l_simple_weight=1.,
                 l_elbo_weight=0.,
                 
                 v_posterior=0.,
                 learn_logvar=False, 
                 logvar_init=0, 

                 latent_scale_factor=None,):

        super().__init__()
        assert parameterization in ["eps", "x0"], \
            'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        highlight_print("Running in {} mode".format(self.parameterization))

        self.vae = self.get_model_list(vae_cfg_list)
        self.ctx = self.get_model_list(ctx_cfg_list)
        self.diffuser = self.get_model_list(diffuser_cfg_list)
        self.global_layer_ptr = global_layer_ptr

        assert self.check_diffuser(), 'diffuser layers are not aligned!'

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print_log(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.loss_type = loss_type
        self.l_simple_weight = l_simple_weight
        self.l_elbo_weight = l_elbo_weight
        self.v_posterior = v_posterior
        self.device = 'cpu'

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=beta_linear_start,
            linear_end=beta_linear_end,
            cosine_s=cosine_s)

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.latent_scale_factor = {} if latent_scale_factor is None else latent_scale_factor

        self.parameter_group = {}
        for namei, diffuseri in self.diffuser.items():
            self.parameter_group.update({
                'diffuser_{}_{}'.format(namei, pgni):pgi for pgni, pgi in diffuseri.parameter_group.items()
            })

    def to(self, device):
        self.device = device
        super().to(device)

    def get_model_list(self, cfg_list):
        net = nn.ModuleDict()
        for name, cfg in cfg_list:
            if not isinstance(cfg, str):
                net[name] = get_model()(cfg)
            else:
                net[name] = String_Reg_Buffer(cfg)
        return net

    def register_schedule(self, 
                          given_betas=None, 
                          beta_schedule="linear", 
                          timesteps=1000,
                          linear_start=1e-4, 
                          linear_end=2e-2, 
                          cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, \
            'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print_log(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print_log(f"{context}: Restored training weights")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        value1 = extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        value2 = extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return value1*x_t -value2*noise

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def forward(self, x_info, c_info):
        x = x_info['x']
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x_info, t, c_info)

    def p_losses(self, x_info, t, c_info, noise=None):
        x = x_info['x']
        noise = torch.randn_like(x) if noise is None else noise
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        x_info['x'] = x_noisy
        model_output = self.apply_model(x_info, t, c_info)

        loss_dict = {}

        if self.parameterization == "x0":
            target = x
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        bs = model_output.shape[0]
        loss_simple = self.get_loss(model_output, target, mean=False).view(bs, -1).mean(-1)
        loss_dict['loss_simple'] = loss_simple.mean()

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict['loss_gamma'] = loss.mean()
            loss_dict['logvar'    ] = self.logvar.data.mean()

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).view(bs, -1).mean(-1)
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict['loss_vlb'] = loss_vlb
        loss_dict.update({'Loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def vae_encode(self, x, which, **kwargs):
        z = self.vae[which].encode(x, **kwargs)
        if self.latent_scale_factor is not None:
            if self.latent_scale_factor.get(which, None) is not None:
                scale = self.latent_scale_factor[which]
                return scale * z
        return z

    @torch.no_grad()
    def vae_decode(self, z, which, **kwargs):
        if self.latent_scale_factor is not None:
            if self.latent_scale_factor.get(which, None) is not None:
                scale = self.latent_scale_factor[which]
                z = 1./scale * z
        x = self.vae[which].decode(z, **kwargs)
        return x

    @torch.no_grad()
    def ctx_encode(self, x, which, **kwargs):
        if which.find('vae_') == 0:
            return self.vae[which[4:]].encode(x, **kwargs)
        else:
            return self.ctx[which].encode(x, **kwargs)

    def ctx_encode_trainable(self, x, which, **kwargs):
        if which.find('vae_') == 0:
            return self.vae[which[4:]].encode(x, **kwargs)
        else:
            return self.ctx[which].encode(x, **kwargs)

    def check_diffuser(self):
        for idx, (_, diffuseri) in enumerate(self.diffuser.items()):
            if idx==0:
                order = diffuseri.layer_order
            else:
                if not order == diffuseri.layer_order:
                    return False
        return True

    @torch.no_grad()
    def on_train_batch_start(self, x):
        pass

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def apply_model(self, x_info, timesteps, c_info):
        x_type, x = x_info['type'], x_info['x']
        c_type, c = c_info['type'], c_info['c']
        dtype = x.dtype

        hs = []

        from .openaimodel import timestep_embedding

        glayer_ptr = x_type if self.global_layer_ptr is None else self.global_layer_ptr
        model_channels = self.diffuser[glayer_ptr].model_channels
        t_emb = timestep_embedding(timesteps, model_channels, repeat_only=False).to(dtype)
        emb = self.diffuser[glayer_ptr].time_embed(t_emb)

        d_iter = iter(self.diffuser[x_type].data_blocks)
        c_iter = iter(self.diffuser[c_type].context_blocks)

        i_order = self.diffuser[x_type].i_order
        m_order = self.diffuser[x_type].m_order
        o_order = self.diffuser[x_type].o_order

        h = x
        for ltype in i_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, c)
            elif ltype == 'save_hidden_feature':
                hs.append(h)

        for ltype in m_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, c)

        for ltype in o_order:
            if ltype == 'load_hidden_feature':
                h = torch.cat([h, hs.pop()], dim=1)
            elif ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, c)
        o = h

        return o

    def context_mixing(self, x, emb, context_module_list, context_info_list, mixing_type):
        nm = len(context_module_list)
        nc = len(context_info_list)
        assert nm == nc
        context = [c_info['c'] for c_info in context_info_list]
        cratio = np.array([c_info['ratio'] for c_info in context_info_list])
        cratio = cratio / cratio.sum()

        if mixing_type == 'attention':
            h = None
            for module, c, r in zip(context_module_list, context, cratio):
                hi = module(x, emb, c) * r
                h = h+hi if h is not None else hi
            return h
        elif mixing_type == 'layer':
            ni = npr.choice(nm, p=cratio)
            module = context_module_list[ni]
            c = context[ni]
            h = module(x, emb, c)
            return h

    def apply_model_multicontext(self, x_info, timesteps, c_info_list, mixing_type='attention'):
        '''
        context_info_list: [[context_type, context, ratio]] for 'attention'
        '''

        x_type, x = x_info['type'], x_info['x']
        dtype = x.dtype

        hs = []

        from .openaimodel import timestep_embedding
        model_channels = self.diffuser[x_type].model_channels
        t_emb = timestep_embedding(timesteps, model_channels, repeat_only=False).to(dtype)
        emb = self.diffuser[x_type].time_embed(t_emb)

        d_iter = iter(self.diffuser[x_type].data_blocks)
        c_iter_list = [iter(self.diffuser[c_info['type']].context_blocks) for c_info in c_info_list]

        i_order = self.diffuser[x_type].i_order
        m_order = self.diffuser[x_type].m_order
        o_order = self.diffuser[x_type].o_order

        h = x
        for ltype in i_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module_list = [next(c_iteri) for c_iteri in c_iter_list]
                h = self.context_mixing(h, emb, module_list, c_info_list, mixing_type)
            elif ltype == 'save_hidden_feature':
                hs.append(h)

        for ltype in m_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module_list = [next(c_iteri) for c_iteri in c_iter_list]
                h = self.context_mixing(h, emb, module_list, c_info_list, mixing_type)

        for ltype in o_order:
            if ltype == 'load_hidden_feature':
                h = torch.cat([h, hs.pop()], dim=1)
            elif ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module_list = [next(c_iteri) for c_iteri in c_iter_list]
                h = self.context_mixing(h, emb, module_list, c_info_list, mixing_type)
        o = h
        return o
