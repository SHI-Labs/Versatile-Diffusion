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

version = '0'
symbol = 'vd'

from .diffusion_utils import \
    count_params, extract_into_tensor, make_beta_schedule
from .distributions import normal_kl, DiagonalGaussianDistribution

from .autoencoder import AutoencoderKL
from .ema import LitEma

from .sd import highlight_print, DDPM, SD_T2I

@register('vd_basic', version)
class VD_Basic(SD_T2I):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def is_part_of_crossattn(name):
            if name.find('.1.norm')!=-1:
                return True
            if name.find('.1.proj_in')!=-1:
                return True
            if name.find('.1.transformer_blocks')!=-1:
                return True
            if name.find('.1.proj_out')!=-1:
                return True
            return False

        self.parameter_group = {
            'context' :[v for n, v in self.model.named_parameters() if is_part_of_crossattn(n)],
            'data'    :[v for n, v in self.model.named_parameters() if not is_part_of_crossattn(n)],
        }

        self.encode_image = None
        self.encode_text = None
        self._predict_eps_from_xstart = None
        self._prior_bpd = None
        self.p_mean_variance = None
        self.p_sample = None
        self.progressive_denoising = None
        self.p_sample_loop = None
        self.sample = None

    @torch.no_grad()
    def encode_input(self, im):
        encoder_posterior = self.first_stage_model.encode(im)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError("Encoder_posterior of type '{}' not yet implemented".format(type(encoder_posterior)))
        return z * self.scale_factor

    @torch.no_grad()
    def decode_latent(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def clip_encode_vision(self, vision, encode_type='encode_vision'):
        clip_encode_type = self.cond_stage_model.encode_type
        self.cond_stage_model.encode_type = encode_type
        if isinstance(vision, torch.Tensor):
            vision = ((vision+1)/2).to('cpu').numpy()
            vision = np.transpose(vision, (0, 2, 3, 1))
            vision = [vi for vi in vision]

        embedding = self.encode_conditioning(vision)
        self.cond_stage_model.encode_type = clip_encode_type
        return embedding

    def encode_conditioning(self, c):
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(c)
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
        else:
            c = self.cond_stage_model(c)
        return c

    # legacy
    def get_learned_conditioning(self, c):
        return self.encode_conditioning(c)

    def forward(self, x, c, noise=None):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        if self.cond_stage_trainable:
            c = self.encode_conditioning(c)
        return self.p_losses(x, c, t, noise)

@register('vd_dc', version)
class VD_DualContext(SD_T2I):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def is_part_of_trans(name):
            if name.find('.1.norm')!=-1:
                return True
            if name.find('.1.proj_in')!=-1:
                return True
            if name.find('.1.transformer_blocks')!=-1:
                return True
            if name.find('.1.proj_out')!=-1:
                return True
            return False

        self.parameter_group = {
            'transformers' : [v for n, v in self.model.named_parameters() if is_part_of_trans(n)],
            'other' :[v for n, v in self.model.named_parameters() if not is_part_of_trans(n)],
        }

    def apply_model(self, x_noisy, t, cond, cond_type):
        if cond_type in ['prompt', 'text']:
            which_attn = 0
        elif cond_type in ['vision', 'visual', 'image']:
            which_attn = 1
        elif isinstance(cond_type, float):
            assert 0 < cond_type < 1, \
                'A special cond_type that will doing a random mix between two input condition, '\
                'rand() < cond_type is text, else visual'
            which_attn = cond_type
        else:
            assert False
        return self.model.diffusion_model(x_noisy, t, cond, which_attn=which_attn)

    def p_losses(self, x_start, cond, t, noise=None, cond_type=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond, cond_type=cond_type)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict['loss_simple'] = loss_simple.mean()

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict['loss_gamma'] = loss.mean()
            loss_dict['logvar'    ] = self.logvar.data.mean()

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict['loss_vlb'] = loss_vlb

        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({'Loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def clip_encode_text(self, text):
        clip_encode_type = self.cond_stage_model.encode_type
        self.cond_stage_model.encode_type = 'encode_text'
        embedding = self.get_learned_conditioning(text)
        self.cond_stage_model.encode_type = clip_encode_type
        return embedding

    @torch.no_grad()
    def clip_encode_vision(self, vision, encode_type='encode_vision'):
        clip_encode_type = self.cond_stage_model.encode_type
        self.cond_stage_model.encode_type = encode_type
        if isinstance(vision, torch.Tensor):
            vision = ((vision+1)/2).to('cpu').numpy()
            vision = np.transpose(vision, (0, 2, 3, 1))
            vision = [vi for vi in vision]
        embedding = self.get_learned_conditioning(vision)
        self.cond_stage_model.encode_type = clip_encode_type
        return embedding

    def get_learned_conditioning(self, c):
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(c)
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
        else:
            c = self.cond_stage_model(c)
        return c

    def forward(self, x, c, noise=None, cond_type=None):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        if self.cond_stage_trainable:
            c = self.get_learned_conditioning(c)
        return self.p_losses(x, c, t, noise, cond_type=cond_type)

@register('vd', version)
class VD(DDPM):
    def __init__(self,
                 autokl_cfg,
                 optimus_cfg,
                 clip_cfg,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, 
                 **kwargs):
        self.scale_by_std = scale_by_std
        super().__init__(*args, **kwargs)

        self.autokl = get_model()(autokl_cfg)
        self.optimus = get_model()(optimus_cfg)
        self.clip = get_model()(clip_cfg)

        self.concat_mode = 'crossattn'
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.device = 'cpu'
        self.parameter_group = self.create_parameter_group()

    def create_parameter_group(self):
        def is_part_of_unet_image(name):
            if name.find('.unet_image.')!=-1:
                return True
            return False
        def is_part_of_unet_text(name):
            if name.find('.unet_text.')!=-1:
                return True
            return False
        def is_part_of_trans(name):
            if name.find('.1.norm')!=-1:
                return True
            if name.find('.1.proj_in')!=-1:
                return True
            if name.find('.1.transformer_blocks')!=-1:
                return True
            if name.find('.1.proj_out')!=-1:
                return True
            return False
        parameter_group = {
            'image_trans' : [],
            'image_rest'  : [],
            'text_trans'  : [],
            'text_rest'   : [],
            'rest'        : [],}
        for pname, para in self.model.named_parameters():
            if is_part_of_unet_image(pname):
                if is_part_of_trans(pname):
                    parameter_group['image_trans'].append(para)
                else:
                    parameter_group['image_rest'].append(para)
            elif is_part_of_unet_text(pname):
                if is_part_of_trans(pname):
                    parameter_group['text_trans'].append(para)
                else:
                    parameter_group['text_rest'].append(para)
            else:
                parameter_group['rest'].append(para)

        return parameter_group

    def to(self, device):
        self.device = device
        super().to(device)

    @torch.no_grad()
    def on_train_batch_start(self, x):
        # only for very first batch
        if self.scale_by_std:
            assert self.scale_factor == 1., \
                'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            highlight_print("setting self.scale_factor to {}".format(self.scale_factor))

    @torch.no_grad()
    def autokl_encode(self, image):
        encoder_posterior = self.autokl.encode(image)
        z = encoder_posterior.sample()
        return self.scale_factor * z

    @torch.no_grad()
    def autokl_decode(self, z):
        z = 1. / self.scale_factor * z
        return self.autokl.decode(z)

    def mask_tokens(inputs, tokenizer, args):
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        
        masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).to(torch.uint8)
        labels[masked_indices==1] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(torch.uint8) & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(torch.uint8) & masked_indices & ~indices_replaced
        indices_random = indices_random
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    @torch.no_grad()
    def optimus_encode(self, text):
        tokenizer = self.optimus.tokenizer_encoder
        token = [tokenizer.tokenize(sentence.lower()) for sentence in text]
        token_id = []
        for tokeni in token:
            token_sentence = [tokenizer._convert_token_to_id(i) for i in tokeni]
            token_sentence = tokenizer.add_special_tokens_single_sentence(token_sentence)
            token_id.append(torch.LongTensor(token_sentence))
        token_id = torch._C._nn.pad_sequence(token_id, batch_first=True, padding_value=0.0)
        token_id = token_id.to(self.device)
        z = self.optimus.encoder(token_id, attention_mask=(token_id > 0).float())[1]
        z_mu, z_logvar = self.optimus.encoder.linear(z).chunk(2, -1)
        # z_sampled = self.optimus.reparameterize(z_mu, z_logvar, 1)
        return z_mu.squeeze(1)

    @torch.no_grad()
    def optimus_decode(self, z, temperature=1.0):
        bos_token = self.optimus.tokenizer_decoder.encode('<BOS>')
        eos_token = self.optimus.tokenizer_decoder.encode('<EOS>')
        context_tokens = torch.LongTensor(bos_token).to(z.device)

        from .optimus import sample_single_sequence_conditional
        sentenses = []
        for zi in z:
            out = sample_single_sequence_conditional(
                model=self.optimus.decoder,
                context=context_tokens,
                past=zi, temperature=temperature, 
                top_k=0, top_p=1.0,
                max_length=30,
                eos_token = eos_token[0],)
            text = self.optimus.tokenizer_decoder.decode(out.tolist(), clean_up_tokenization_spaces=True)
            text = text.split()[1:-1]
            text = ' '.join(text)
            sentenses.append(text)
        return sentenses

    @torch.no_grad()
    def clip_encode_text(self, text, encode_type='encode_text'):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        embedding = self.clip.encode(text)
        self.clip.encode_type = swap_type
        return embedding

    @torch.no_grad()
    def clip_encode_vision(self, vision, encode_type='encode_vision'):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        if isinstance(vision, torch.Tensor):
            vision = ((vision+1)/2).to('cpu').numpy()
            vision = np.transpose(vision, (0, 2, 3, 1))
            vision = [vi for vi in vision]
        embedding = self.clip.encode(vision)
        self.clip.encode_type = swap_type
        return embedding

    def forward(self, x, c, noise=None, xtype='image', ctype='prompt'):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, c, t, noise, xtype, ctype)

    def apply_model(self, x_noisy, t, cond, xtype='image', ctype='prompt'):
        return self.model.diffusion_model(x_noisy, t, cond, xtype, ctype)

    def get_image_loss(self, pred, target, mean=True):
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

    def get_text_loss(self, pred, target):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        return loss

    def p_losses(self, x_start, cond, t, noise=None, xtype='image', ctype='prompt'):
        noise = torch.randn_like(x_start) if noise is None else noise
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond, xtype, ctype)

        loss_dict = {}

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        if xtype == 'image':
            loss_simple = self.get_image_loss(model_output, target, mean=False).mean([1, 2, 3])
        elif xtype == 'text':
            loss_simple = self.get_text_loss(model_output, target).mean([1])

        logvar_t = self.logvar[t].to(self.device)
        if logvar_t.sum().item() != 0:
            assert False, "Default SD training has logvar fixed at 0"
        if self.learn_logvar:
            assert False, "Default SD training don't learn logvar"
        if self.l_simple_weight != 1:
            assert False, "Default SD training always set l_simple_weight==1"

        loss = loss_simple.mean()
        loss_dict['loss_simple'] = loss_simple.mean().item()
        loss_dict['Loss'] = loss.item()
        return loss, loss_dict

    def apply_model_dc(self, x_noisy, t, first_c, second_c, xtype='image', first_ctype='vision', second_ctype='prompt', mixed_ratio=0.5):
        return self.model.diffusion_model.forward_dc(x_noisy, t, first_c, second_c, xtype, first_ctype, second_ctype, mixed_ratio)