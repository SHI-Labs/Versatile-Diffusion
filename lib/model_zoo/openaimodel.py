from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_utils import \
    checkpoint, conv_nd, linear, avg_pool_nd, \
    zero_module, normalization, timestep_embedding

from .attention import SpatialTransformer

from lib.model_zoo.common.get_model import get_model, register

symbol = 'openai'

# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


@register('openai_unet')
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        #self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if disable_self_attentions is not None:
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if num_attention_blocks is None or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if disable_self_attentions is not None:
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if num_attention_blocks is None or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        *args,
        **kwargs
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)


#######################
# Unet with self-attn #
#######################

from .attention import SpatialTransformerNoContext

@register('openai_unet_nocontext')
class UNetModelNoContext(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,    # custom transformer support
            transformer_depth=1,              # custom transformer support
            n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            num_attention_blocks=None, ):

        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        #self.num_res_blocks = num_res_blocks
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformerNoContext(
                                ch, num_heads, dim_head, depth=transformer_depth
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformerNoContext(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformerNoContext(
                                ch, num_heads, dim_head, depth=transformer_depth,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def forward(self, x, timesteps):
        assert self.num_classes is None, \
            "not supported"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

@register('openai_unet_nocontext_noatt')
class UNetModelNoContextNoAtt(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            use_scale_shift_norm=False,
            resblock_updown=False,
            n_embed=None,):

        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        #self.num_res_blocks = num_res_blocks

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def forward(self, x, timesteps):
        assert self.num_classes is None, \
            "not supported"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

@register('openai_unet_nocontext_noatt_decoderonly')
class UNetModelNoContextNoAttDecoderOnly(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            model_channels,
            num_res_blocks,
            dropout=0,
            channel_mult=(4, 2, 1),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            use_scale_shift_norm=False,
            resblock_updown=False,
            n_embed=None,):

        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        #self.num_res_blocks = num_res_blocks

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self._feature_size = model_channels

        ch = model_channels * self.channel_mult[0]
        self.output_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, ch, 3, padding=1)
                )
            ]
        )

        for level, mult in enumerate(channel_mult):
            for i in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if level != len(channel_mult)-1 and (i == self.num_res_blocks[level]-1):
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def forward(self, x, timesteps):
        assert self.num_classes is None, \
            "not supported"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for module in self.output_blocks:
            h = module(h, emb)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

#########################
# Double Attention Unet #
#########################

from .attention import DualSpatialTransformer

class TimestepEmbedSequentialExtended(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, which_attn=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, DualSpatialTransformer):
                x = layer(x, context, which=which_attn)
            else:
                x = layer(x)
        return x

@register('openai_unet_dual_context')
class UNetModelDualContext(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None, ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        #self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequentialExtended(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if disable_self_attentions is not None:
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if num_attention_blocks is None or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else DualSpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequentialExtended(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequentialExtended(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequentialExtended(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else DualSpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if disable_self_attentions is not None:
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if num_attention_blocks is None or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else DualSpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequentialExtended(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def forward(self, x, timesteps=None, context=None, y=None, which_attn=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(context.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context, which_attn=which_attn)
            hs.append(h)
        h = self.middle_block(h, emb, context, which_attn=which_attn)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, which_attn=which_attn)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

###########
# VD Unet #
###########

from functools import partial

@register('openai_unet_2d')
class UNetModel2D(nn.Module):
    def __init__(self,
                 input_channels,
                 model_channels,
                 output_channels,
                 context_dim=768,
                 num_noattn_blocks=(2, 2, 2, 2),
                 channel_mult=(1, 2, 4, 8),
                 with_attn=[True, True, True, False],
                 num_heads=8,
                 use_checkpoint=True, ):

        super().__init__()

        ResBlockPreset = partial(
            ResBlock, dropout=0, dims=2, use_checkpoint=use_checkpoint, 
            use_scale_shift_norm=False)
 
        self.input_channels = input_channels
        self.model_channels = model_channels
        self.num_noattn_blocks = num_noattn_blocks
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        ##################
        # Time embedding #
        ##################

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),)

        ################
        # input_blocks #
        ################
        current_channel = model_channels
        input_blocks = [
            TimestepEmbedSequential(
                nn.Conv2d(input_channels, model_channels, 3, padding=1, bias=True))]
        input_block_channels = [current_channel]

        for level_idx, mult in enumerate(channel_mult):
            for _ in range(self.num_noattn_blocks[level_idx]):
                layers = [
                    ResBlockPreset(
                        current_channel, time_embed_dim,
                        out_channels = mult * model_channels,)]

                current_channel = mult * model_channels
                dim_head = current_channel // num_heads
                if with_attn[level_idx]:
                    layers += [
                        SpatialTransformer(
                            current_channel, num_heads, dim_head, 
                            depth=1, context_dim=context_dim, )]

                input_blocks += [TimestepEmbedSequential(*layers)]
                input_block_channels.append(current_channel)

            if level_idx != len(channel_mult) - 1:
                input_blocks += [
                    TimestepEmbedSequential(
                        Downsample(
                            current_channel, use_conv=True, 
                            dims=2, out_channels=current_channel,))]
                input_block_channels.append(current_channel)

        self.input_blocks = nn.ModuleList(input_blocks)

        #################
        # middle_blocks #
        #################
        middle_block = [
            ResBlockPreset(
                current_channel, time_embed_dim,),
            SpatialTransformer(
                current_channel, num_heads, dim_head, 
                depth=1, context_dim=context_dim, ),
            ResBlockPreset(
                current_channel, time_embed_dim,),]
        self.middle_block = TimestepEmbedSequential(*middle_block)

        #################
        # output_blocks #
        #################
        output_blocks = []
        for level_idx, mult in list(enumerate(channel_mult))[::-1]:
            for block_idx in range(self.num_noattn_blocks[level_idx] + 1):
                extra_channel = input_block_channels.pop()
                layers = [
                    ResBlockPreset(
                        current_channel + extra_channel,
                        time_embed_dim,
                        out_channels = model_channels * mult,) ]

                current_channel = model_channels * mult
                dim_head = current_channel // num_heads
                if with_attn[level_idx]:
                    layers += [
                        SpatialTransformer(
                            current_channel, num_heads, dim_head, 
                            depth=1, context_dim=context_dim,)]

                if level_idx!=0 and block_idx==self.num_noattn_blocks[level_idx]:
                    layers += [
                        Upsample(
                            current_channel, use_conv=True, 
                            dims=2, out_channels=current_channel)]
 
                output_blocks += [TimestepEmbedSequential(*layers)]

        self.output_blocks = nn.ModuleList(output_blocks)

        self.out = nn.Sequential(
            normalization(current_channel),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, output_channels, 3, padding=1)),)

    def forward(self, x, timesteps=None, context=None):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)

class FCBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 1, padding=0),)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, self.out_channels,),)
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 1, padding=0)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1, padding=0)

    def forward(self, x, emb):
        if len(x.shape) == 2:
            x = x[:, :, None, None]
        elif len(x.shape) == 4:
            pass
        else:
            raise ValueError
        y = checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        if len(x.shape) == 2:
            return y[:, :, 0, 0]
        elif len(x.shape) == 4:
            return y

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

@register('openai_unet_0d')
class UNetModel0D(nn.Module):
    def __init__(self,
                 input_channels,
                 model_channels,
                 output_channels,
                 context_dim=768,
                 num_noattn_blocks=(2, 2, 2, 2),
                 channel_mult=(1, 2, 4, 8),
                 with_attn=[True, True, True, False],
                 num_heads=8,
                 use_checkpoint=True, ):

        super().__init__()

        FCBlockPreset = partial(FCBlock, dropout=0, use_checkpoint=use_checkpoint)
 
        self.input_channels = input_channels
        self.model_channels = model_channels
        self.num_noattn_blocks = num_noattn_blocks
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        ##################
        # Time embedding #
        ##################

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),)

        ################
        # input_blocks #
        ################
        current_channel = model_channels
        input_blocks = [
            TimestepEmbedSequential(
                nn.Conv2d(input_channels, model_channels, 1, padding=0, bias=True))]
        input_block_channels = [current_channel]

        for level_idx, mult in enumerate(channel_mult):
            for _ in range(self.num_noattn_blocks[level_idx]):
                layers = [
                    FCBlockPreset(
                        current_channel, time_embed_dim,
                        out_channels = mult * model_channels,)]

                current_channel = mult * model_channels
                dim_head = current_channel // num_heads
                if with_attn[level_idx]:
                    layers += [
                        SpatialTransformer(
                            current_channel, num_heads, dim_head, 
                            depth=1, context_dim=context_dim, )]

                input_blocks += [TimestepEmbedSequential(*layers)]
                input_block_channels.append(current_channel)

            if level_idx != len(channel_mult) - 1:
                input_blocks += [
                    TimestepEmbedSequential(
                        Downsample(
                            current_channel, use_conv=True, 
                            dims=2, out_channels=current_channel,))]
                input_block_channels.append(current_channel)

        self.input_blocks = nn.ModuleList(input_blocks)

        #################
        # middle_blocks #
        #################
        middle_block = [
            FCBlockPreset(
                current_channel, time_embed_dim,),
            SpatialTransformer(
                current_channel, num_heads, dim_head, 
                depth=1, context_dim=context_dim, ),
            FCBlockPreset(
                current_channel, time_embed_dim,),]
        self.middle_block = TimestepEmbedSequential(*middle_block)

        #################
        # output_blocks #
        #################
        output_blocks = []
        for level_idx, mult in list(enumerate(channel_mult))[::-1]:
            for block_idx in range(self.num_noattn_blocks[level_idx] + 1):
                extra_channel = input_block_channels.pop()
                layers = [
                    FCBlockPreset(
                        current_channel + extra_channel,
                        time_embed_dim,
                        out_channels = model_channels * mult,) ]

                current_channel = model_channels * mult
                dim_head = current_channel // num_heads
                if with_attn[level_idx]:
                    layers += [
                        SpatialTransformer(
                            current_channel, num_heads, dim_head, 
                            depth=1, context_dim=context_dim,)]

                if level_idx!=0 and block_idx==self.num_noattn_blocks[level_idx]:
                    layers += [
                        nn.Conv2d(current_channel, current_channel, 1, padding=0)]

                output_blocks += [TimestepEmbedSequential(*layers)]

        self.output_blocks = nn.ModuleList(output_blocks)

        self.out = nn.Sequential(
            normalization(current_channel),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, output_channels, 1, padding=0)),)

    def forward(self, x, timesteps=None, context=None):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)

class Linear_MultiDim(nn.Linear):
    def __init__(self, in_features, out_features, *args, **kwargs):
        
        in_features = [in_features] if isinstance(in_features, int) else list(in_features)
        out_features = [out_features] if isinstance(out_features, int) else list(out_features)
        self.in_features_multidim = in_features
        self.out_features_multidim = out_features
        super().__init__(
            np.array(in_features).prod(), 
            np.array(out_features).prod(), 
            *args, **kwargs)

    def forward(self, x):
        shape = x.shape
        n = len(shape) - len(self.in_features_multidim)
        x = x.view(*shape[:n], self.in_features)
        y = super().forward(x)
        y = y.view(*shape[:n], *self.out_features_multidim)
        return y

class FCBlock_MultiDim(FCBlock):
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_checkpoint=False,):
        channels = [channels] if isinstance(channels, int) else list(channels)
        channels_all = np.array(channels).prod()
        self.channels_multidim = channels

        if out_channels is not None:
            out_channels = [out_channels] if isinstance(out_channels, int) else list(out_channels)
            out_channels_all = np.array(out_channels).prod()
            self.out_channels_multidim = out_channels
        else:
            out_channels_all = channels_all
            self.out_channels_multidim = self.channels_multidim

        self.channels = channels
        super().__init__(
            channels = channels_all,
            emb_channels = emb_channels,
            dropout = dropout,
            out_channels = out_channels_all,
            use_checkpoint = use_checkpoint,)

    def forward(self, x, emb):
        shape = x.shape
        n = len(self.channels_multidim)
        x = x.view(*shape[0:-n], self.channels, 1, 1)
        x = x.view(-1, self.channels, 1, 1)
        y = checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        y = y.view(*shape[0:-n], -1)
        y = y.view(*shape[0:-n], *self.out_channels_multidim)
        return y

@register('openai_unet_0dmd')
class UNetModel0D_MultiDim(nn.Module):
    def __init__(self,
                 input_channels,
                 model_channels,
                 output_channels,
                 context_dim=768,
                 num_noattn_blocks=(2, 2, 2, 2),
                 channel_mult=(1, 2, 4, 8),
                 second_dim=(4, 4, 4, 4),
                 with_attn=[True, True, True, False],
                 num_heads=8,
                 use_checkpoint=True, ):

        super().__init__()

        FCBlockPreset = partial(FCBlock_MultiDim, dropout=0, use_checkpoint=use_checkpoint)
 
        self.input_channels = input_channels
        self.model_channels = model_channels
        self.num_noattn_blocks = num_noattn_blocks
        self.channel_mult = channel_mult
        self.second_dim = second_dim
        self.num_heads = num_heads

        ##################
        # Time embedding #
        ##################

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),)

        ################
        # input_blocks #
        ################
        sdim = second_dim[0]
        current_channel = [model_channels, sdim, 1]
        input_blocks = [
            TimestepEmbedSequential(
                Linear_MultiDim([input_channels, 1, 1], current_channel, bias=True))]
        input_block_channels = [current_channel]

        for level_idx, (mult, sdim) in enumerate(zip(channel_mult, second_dim)):
            for _ in range(self.num_noattn_blocks[level_idx]):
                layers = [
                    FCBlockPreset(
                        current_channel, 
                        time_embed_dim,
                        out_channels = [mult*model_channels, sdim, 1],)]

                current_channel = [mult*model_channels, sdim, 1]
                dim_head = current_channel[0] // num_heads
                if with_attn[level_idx]:
                    layers += [
                        SpatialTransformer(
                            current_channel[0], num_heads, dim_head, 
                            depth=1, context_dim=context_dim, )]

                input_blocks += [TimestepEmbedSequential(*layers)]
                input_block_channels.append(current_channel)

            if level_idx != len(channel_mult) - 1:
                input_blocks += [
                    TimestepEmbedSequential(
                        Linear_MultiDim(current_channel, current_channel, bias=True, ))]
                input_block_channels.append(current_channel)

        self.input_blocks = nn.ModuleList(input_blocks)

        #################
        # middle_blocks #
        #################
        middle_block = [
            FCBlockPreset(
                current_channel, time_embed_dim, ),
            SpatialTransformer(
                current_channel[0], num_heads, dim_head, 
                depth=1, context_dim=context_dim, ),
            FCBlockPreset(
                current_channel, time_embed_dim, ),]
        self.middle_block = TimestepEmbedSequential(*middle_block)

        #################
        # output_blocks #
        #################
        output_blocks = []
        for level_idx, (mult, sdim) in list(enumerate(zip(channel_mult, second_dim)))[::-1]:
            for block_idx in range(self.num_noattn_blocks[level_idx] + 1):
                extra_channel = input_block_channels.pop()
                layers = [
                    FCBlockPreset(
                        [current_channel[0] + extra_channel[0]] + current_channel[1:],
                        time_embed_dim,
                        out_channels = [mult*model_channels, sdim, 1], )]

                current_channel = [mult*model_channels, sdim, 1]
                dim_head = current_channel[0] // num_heads
                if with_attn[level_idx]:
                    layers += [
                        SpatialTransformer(
                            current_channel[0], num_heads, dim_head, 
                            depth=1, context_dim=context_dim,)]

                if level_idx!=0 and block_idx==self.num_noattn_blocks[level_idx]:
                    layers += [
                        Linear_MultiDim(current_channel, current_channel, bias=True, )]

                output_blocks += [TimestepEmbedSequential(*layers)]

        self.output_blocks = nn.ModuleList(output_blocks)

        self.out = nn.Sequential(
            normalization(current_channel[0]),
            nn.SiLU(),
            zero_module(Linear_MultiDim(current_channel, [output_channels, 1, 1], bias=True, )),)

    def forward(self, x, timesteps=None, context=None):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)

@register('openai_unet_vd')
class UNetModelVD(nn.Module):
    def __init__(self,
                 unet_image_cfg,  
                 unet_text_cfg, ):

        super().__init__()
        self.unet_image = get_model()(unet_image_cfg)
        self.unet_text = get_model()(unet_text_cfg)
        self.time_embed = self.unet_image.time_embed
        del self.unet_image.time_embed
        del self.unet_text.time_embed

        self.model_channels = self.unet_image.model_channels
        
    def forward(self, x, timesteps, context, xtype='image', ctype='prompt'):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb.to(x.dtype))

        if xtype == 'text':
            x = x[:, :, None, None]

        h = x
        for i_module, t_module in zip(self.unet_image.input_blocks, self.unet_text.input_blocks):
            h = self.mixed_run(i_module, t_module, h, emb, context, xtype, ctype)
            hs.append(h)
        h = self.mixed_run(
            self.unet_image.middle_block, self.unet_text.middle_block, 
            h, emb, context, xtype, ctype)
        for i_module, t_module in zip(self.unet_image.output_blocks, self.unet_text.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = self.mixed_run(i_module, t_module, h, emb, context, xtype, ctype)
        if xtype == 'image':
            return self.unet_image.out(h)
        elif xtype == 'text':
            return self.unet_text.out(h).squeeze(-1).squeeze(-1)

    def mixed_run(self, inet, tnet, x, emb, context, xtype, ctype):

        h = x
        for ilayer, tlayer in zip(inet, tnet):
            if isinstance(ilayer, TimestepBlock) and xtype=='image':
                h = ilayer(h, emb)
            elif isinstance(tlayer, TimestepBlock) and xtype=='text':
                h = tlayer(h, emb)
            elif isinstance(ilayer, SpatialTransformer) and ctype=='vision':
                h = ilayer(h, context)
            elif isinstance(ilayer, SpatialTransformer) and ctype=='prompt':
                h = tlayer(h, context)
            elif xtype=='image':
                h = ilayer(h)
            elif xtype == 'text':
                h = tlayer(h)
            else:
                raise ValueError
        return h

    def forward_dc(self, x, timesteps, c0, c1, xtype, c0_type, c1_type, mixed_ratio):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb.to(x.dtype))

        if xtype == 'text':
            x = x[:, :, None, None]
        h = x
        for i_module, t_module in zip(self.unet_image.input_blocks, self.unet_text.input_blocks):
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
            hs.append(h)
        h = self.mixed_run_dc(
            self.unet_image.middle_block, self.unet_text.middle_block, 
            h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
        for i_module, t_module in zip(self.unet_image.output_blocks, self.unet_text.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
        if xtype == 'image':
            return self.unet_image.out(h)
        elif xtype == 'text':
            return self.unet_text.out(h).squeeze(-1).squeeze(-1)

    def mixed_run_dc(self, inet, tnet, x, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio):
        h = x
        for ilayer, tlayer in zip(inet, tnet):
            if isinstance(ilayer, TimestepBlock) and xtype=='image':
                h = ilayer(h, emb)
            elif isinstance(tlayer, TimestepBlock) and xtype=='text':
                h = tlayer(h, emb)
            elif isinstance(ilayer, SpatialTransformer):
                h0 = ilayer(h, c0)-h if c0_type=='vision' else tlayer(h, c0)-h
                h1 = ilayer(h, c1)-h if c1_type=='vision' else tlayer(h, c1)-h
                h = h0*mixed_ratio + h1*(1-mixed_ratio) + h
                # h = ilayer(h, c0)
            elif xtype=='image':
                h = ilayer(h)
            elif xtype == 'text':
                h = tlayer(h)
            else:
                raise ValueError
        return h

################
# VD Next Unet #
################

from functools import partial
import copy

@register('openai_unet_2d_next')
class UNetModel2D_Next(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            context_dim,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            use_checkpoint=False,
            num_heads=8,
            num_head_channels=None,
            parts = ['global', 'data', 'context']):

        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.context_dim = context_dim
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        assert (num_heads is None) + (num_head_channels is None) == 1, \
            "One of num_heads or num_head_channels need to be set"

        self.parts = parts if isinstance(parts, list) else [parts]
        self.glayer_included = 'global' in self.parts
        self.dlayer_included = 'data' in self.parts
        self.clayer_included = 'context' in self.parts
        self.layer_sequence_ordering = []

        #################
        # global layers #
        #################

        time_embed_dim = model_channels * 4
        if self.glayer_included:
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        ################
        # input layers #
        ################

        if self.dlayer_included:
            self.data_blocks = nn.ModuleList([])
            ResBlockDefault = partial(
                ResBlock, 
                emb_channels=time_embed_dim,
                dropout=dropout,
                dims=2,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=False, )
        else:
            def dummy(*args, **kwargs):
                return None
            ResBlockDefault = dummy

        if self.clayer_included:
            self.context_blocks = nn.ModuleList([])
            CrossAttnDefault = partial(
                SpatialTransformer, 
                context_dim=context_dim,
                disable_self_attn=False, )
        else:
            def dummy(*args, **kwargs):
                return None
            CrossAttnDefault = dummy

        self.add_data_layer(conv_nd(2, in_channels, model_channels, 3, padding=1))
        self.layer_sequence_ordering.append('save_hidden_feature')
        input_block_chans = [model_channels]

        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layer = ResBlockDefault(
                    channels=ch, out_channels=mult*model_channels,)
                self.add_data_layer(layer)
                ch = mult * model_channels

                if (ds in attention_resolutions):
                    d_head, n_heads = self.get_d_head_n_heads(ch)
                    layer = CrossAttnDefault(
                        in_channels=ch, d_head=d_head, n_heads=n_heads,)
                    self.add_context_layer(layer)
                input_block_chans.append(ch)
                self.layer_sequence_ordering.append('save_hidden_feature')

            if level != len(channel_mult) - 1:
                layer = Downsample(
                    ch, use_conv=True, dims=2, out_channels=ch)
                self.add_data_layer(layer)
                input_block_chans.append(ch)
                self.layer_sequence_ordering.append('save_hidden_feature')
                ds *= 2

        self.i_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        #################
        # middle layers #
        #################

        self.add_data_layer(ResBlockDefault(channels=ch))
        d_head, n_heads = self.get_d_head_n_heads(ch)
        self.add_context_layer(CrossAttnDefault(in_channels=ch, d_head=d_head, n_heads=n_heads))
        self.add_data_layer(ResBlockDefault(channels=ch))

        self.m_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        #################
        # output layers #
        #################

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for _ in range(self.num_res_blocks[level] + 1):
                self.layer_sequence_ordering.append('load_hidden_feature')
                ich = input_block_chans.pop()
                layer = ResBlockDefault(
                    channels=ch+ich, out_channels=model_channels*mult,)
                ch = model_channels * mult
                self.add_data_layer(layer)

                if ds in attention_resolutions:
                    d_head, n_heads = self.get_d_head_n_heads(ch)
                    layer = CrossAttnDefault(
                        in_channels=ch, d_head=d_head, n_heads=n_heads)
                    self.add_context_layer(layer)

            if level != 0:
                layer = Upsample(ch, conv_resample, dims=2, out_channels=ch)
                self.add_data_layer(layer)
                ds //= 2                

        layer = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(2, model_channels, out_channels, 3, padding=1)),
        )
        self.add_data_layer(layer)

        self.o_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_order = copy.deepcopy(self.i_order + self.m_order + self.o_order)
        del self.layer_sequence_ordering

        self.parameter_group = {}
        if self.glayer_included:
            self.parameter_group['global'] = self.time_embed
        if self.dlayer_included:
            self.parameter_group['data'] = self.data_blocks
        if self.clayer_included:
            self.parameter_group['context'] = self.context_blocks

    def get_d_head_n_heads(self, ch):
        if self.num_head_channels is None:
            d_head = ch // self.num_heads
            n_heads = self.num_heads
        else:
            d_head = self.num_head_channels
            n_heads = ch // self.num_head_channels
        return d_head, n_heads

    def add_data_layer(self, layer):
        if self.dlayer_included:
            if not isinstance(layer, (list, tuple)):
                layer = [layer]
            self.data_blocks.append(TimestepEmbedSequential(*layer))
        self.layer_sequence_ordering.append('d')

    def add_context_layer(self, layer):
        if self.clayer_included:
            if not isinstance(layer, (list, tuple)):
                layer = [layer]
            self.context_blocks.append(TimestepEmbedSequential(*layer))
        self.layer_sequence_ordering.append('c')

    def forward(self, x, timesteps, context):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        d_iter = iter(self.data_blocks)
        c_iter = iter(self.context_blocks)

        h = x
        for ltype in self.i_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, context)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, context)
            elif ltype == 'save_hidden_feature':
                hs.append(h)

        for ltype in self.m_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, context)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, context)

        for ltype in self.i_order:
            if ltype == 'load_hidden_feature':
                h = th.cat([h, hs.pop()], dim=1)
            elif ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, context)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, context)
        o = h

        return o

@register('openai_unet_0d_next')
class UNetModel0D_Next(UNetModel2D_Next):
    def __init__(
            self,
            input_channels,
            model_channels,
            output_channels,
            context_dim = 788,
            num_noattn_blocks=(2, 2, 2, 2),
            channel_mult=(1, 2, 4, 8),
            second_dim=(4, 4, 4, 4),
            with_attn=[True, True, True, False],
            num_heads=8,
            num_head_channels=None,
            use_checkpoint=False,
            parts = ['global', 'data', 'context']):

        super(UNetModel2D_Next, self).__init__()

        self.input_channels = input_channels
        self.model_channels = model_channels
        self.output_channels = output_channels
        self.num_noattn_blocks = num_noattn_blocks
        self.channel_mult = channel_mult
        self.second_dim = second_dim
        self.with_attn = with_attn
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        self.parts = parts if isinstance(parts, list) else [parts]
        self.glayer_included = 'global' in self.parts
        self.dlayer_included = 'data' in self.parts
        self.clayer_included = 'context' in self.parts
        self.layer_sequence_ordering = []

        #################
        # global layers #
        #################

        time_embed_dim = model_channels * 4
        if self.glayer_included:
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        ################
        # input layers #
        ################

        if self.dlayer_included:
            self.data_blocks = nn.ModuleList([])
            FCBlockDefault = partial(
                FCBlock_MultiDim, dropout=0, use_checkpoint=use_checkpoint)
        else:
            def dummy(*args, **kwargs):
                return None
            FCBlockDefault = dummy

        if self.clayer_included:
            self.context_blocks = nn.ModuleList([])
            CrossAttnDefault = partial(
                SpatialTransformer, 
                context_dim=context_dim,
                disable_self_attn=False, )
        else:
            def dummy(*args, **kwargs):
                return None
            CrossAttnDefault = dummy

        sdim = second_dim[0]
        current_channel = [model_channels, sdim, 1]
        one_layer = Linear_MultiDim([input_channels], current_channel, bias=True)
        self.add_data_layer(one_layer)
        self.layer_sequence_ordering.append('save_hidden_feature')
        input_block_channels = [current_channel]

        for level_idx, (mult, sdim) in enumerate(zip(channel_mult, second_dim)):
            for _ in range(self.num_noattn_blocks[level_idx]):
                layer = FCBlockDefault(
                    current_channel, 
                    time_embed_dim,
                    out_channels = [mult*model_channels, sdim, 1],)

                self.add_data_layer(layer)
                current_channel = [mult*model_channels, sdim, 1]

                if with_attn[level_idx]:
                    d_head, n_heads = self.get_d_head_n_heads(current_channel[0])
                    layer = CrossAttnDefault(
                        in_channels=current_channel[0],
                        d_head=d_head, n_heads=n_heads,)
                    self.add_context_layer(layer)

                input_block_channels.append(current_channel)
                self.layer_sequence_ordering.append('save_hidden_feature')

            if level_idx != len(channel_mult) - 1:
                layer = Linear_MultiDim(current_channel, current_channel, bias=True,)
                self.add_data_layer(layer)
                input_block_channels.append(current_channel)
                self.layer_sequence_ordering.append('save_hidden_feature')

        self.i_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        #################
        # middle layers #
        #################

        self.add_data_layer(FCBlockDefault(current_channel, time_embed_dim, ))
        d_head, n_heads = self.get_d_head_n_heads(current_channel[0])
        self.add_context_layer(CrossAttnDefault(in_channels=current_channel[0], d_head=d_head, n_heads=n_heads))
        self.add_data_layer(FCBlockDefault(current_channel, time_embed_dim, ))

        self.m_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        #################
        # output layers #
        #################
        for level_idx, (mult, sdim) in list(enumerate(zip(channel_mult, second_dim)))[::-1]:
            for _ in range(self.num_noattn_blocks[level_idx] + 1):
                self.layer_sequence_ordering.append('load_hidden_feature')
                extra_channel = input_block_channels.pop()
                layer = FCBlockDefault(
                    [current_channel[0] + extra_channel[0]] + current_channel[1:],
                    time_embed_dim,
                    out_channels = [mult*model_channels, sdim, 1], )

                self.add_data_layer(layer)
                current_channel = [mult*model_channels, sdim, 1]

                if with_attn[level_idx]:
                    d_head, n_heads = self.get_d_head_n_heads(current_channel[0])
                    layer = CrossAttnDefault(
                        in_channels=current_channel[0], d_head=d_head, n_heads=n_heads)
                    self.add_context_layer(layer)

            if level_idx != 0:
                layer = Linear_MultiDim(current_channel, current_channel, bias=True, )
                self.add_data_layer(layer)

        layer = nn.Sequential(
            normalization(current_channel[0]),
            nn.SiLU(),
            zero_module(Linear_MultiDim(current_channel, [output_channels], bias=True, )),
        )
        self.add_data_layer(layer)

        self.o_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_order = copy.deepcopy(self.i_order + self.m_order + self.o_order)
        del self.layer_sequence_ordering

        self.parameter_group = {}
        if self.glayer_included:
            self.parameter_group['global'] = self.time_embed
        if self.dlayer_included:
            self.parameter_group['data'] = self.data_blocks
        if self.clayer_included:
            self.parameter_group['context'] = self.context_blocks
