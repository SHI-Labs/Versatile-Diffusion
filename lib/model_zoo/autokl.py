import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from lib.model_zoo.common.get_model import get_model, register

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from .autokl_modules import Encoder, Decoder
from .distributions import DiagonalGaussianDistribution

from .autokl_utils import LPIPSWithDiscriminator

@register('autoencoderkl')
class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        if lossconfig is not None:
            self.loss = LPIPSWithDiscriminator(**lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    @torch.no_grad()
    def encode(self, x, out_posterior=False):
        return self.encode_trainable(x, out_posterior)

    def encode_trainable(self, x, out_posterior=False):
        x = x*2-1
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if out_posterior:
            return posterior
        else:
            return posterior.sample()

    @torch.no_grad()
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = torch.clamp((dec+1)/2, 0, 1)
        return dec

    def decode_trainable(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = (dec+1)/2
        return dec

    def apply_model(self, input, sample_posterior=True):
        posterior = self.encode_trainable(input, out_posterior=True)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode_trainable(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def forward(self, x, optimizer_idx, global_step):
        reconstructions, posterior = self.apply_model(x)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(x, reconstructions, posterior, optimizer_idx, global_step=global_step,
                                            last_layer=self.get_last_layer(), split="train")
            return aeloss, log_dict_ae

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(x, reconstructions, posterior, optimizer_idx, global_step=global_step,
                                                last_layer=self.get_last_layer(), split="train")

            return discloss, log_dict_disc

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
