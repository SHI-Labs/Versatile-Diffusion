import os
import os.path as osp
import torch
from collections import OrderedDict
import re

# pthin  = 'pretrained/sd-v1-5-hf.pth'
# pthout = 'pretrained/sd-v1-5.pth'

pthin  = 'pretrained/sd-v1-5-hf.pth'
pthout = 'pretrained/sd-v1-5.pth'

if osp.splitext(pthin)[1] == '.pth':
    sd = torch.load(pthin)
elif osp.splitext(pthin)[1] == '.ckpt':
    sd = torch.load(pthin)['state_dict']
else:
    raise ValueError

operation = 'unet'

# sd2 = torch.load(pthout)
# with open('/home/james/Project/vd-common/log/unet-new.txt', 'w') as f:
#     for ki, vi in sd2.items():
#         f.write('{} | {} | {}\n'.format(ki, vi.shape, vi.abs().sum()))

if operation == 'vae':
    mapping = []

    def move_first_several_layers():
        mapping.append(['encoder.conv_in.weight', 'encoder.conv_in.weight'])
        mapping.append(['encoder.conv_in.bias'  , 'encoder.conv_in.bias'  ])

    def move_down_blocks():
        for k in sd.keys():
            k0 = k.replace('encoder.down_blocks.', 'encoder.down.')
            if k == k0:
                continue
            k0 = k0.replace('.resnets.', '.block.')
            k0 = k0.replace('.downsamplers.0.', '.downsample.')
            k0 = k0.replace('.conv_shortcut.', '.nin_shortcut.')
            mapping.append([k0, k])

    def move_default(ref, to):
        for k in sd.keys():
            k0 = k.replace(ref, to)
            if k == k0:
                continue
            mapping.append([k0, k])

    def move_attention(ref, to):
        for k in sd.keys():
            k0 = k.replace(ref, to)
            if k == k0:
                continue
            k0 = k0.replace('.group_norm.', '.norm.')
            k0 = k0.replace('.query.', '.q.')
            k0 = k0.replace('.key.', '.k.')
            k0 = k0.replace('.value.', '.v.')
            k0 = k0.replace('.proj_attn.', '.proj_out.')
            if (k0.find('.weight')!=-1) and (k0.find('.norm.')==-1):
                f = lambda x: x[:, :, None, None].contiguous()
            else:
                f = lambda x: x
            mapping.append([k0, k, f])

    def move_mid_several_layers():
        mapping.append(['encoder.norm_out.weight', 'encoder.conv_norm_out.weight'])
        mapping.append(['encoder.norm_out.bias',   'encoder.conv_norm_out.bias'  ])
        mapping.append(['encoder.conv_out.weight', 'encoder.conv_out.weight'     ])
        mapping.append(['encoder.conv_out.bias',   'encoder.conv_out.bias'       ])
        mapping.append(['decoder.conv_in.weight',  'decoder.conv_in.weight'      ])
        mapping.append(['decoder.conv_in.bias',    'decoder.conv_in.bias'        ])

    def move_up_blocks(ref, to):
        for k in sd.keys():
            k0 = k.replace('decoder.up_blocks.{}'.format(ref), 'decoder.up.{}'.format(to))
            if k == k0:
                continue
            k0 = k0.replace('.resnets.', '.block.')
            k0 = k0.replace('.upsamplers.0.', '.upsample.')
            k0 = k0.replace('.conv_shortcut.', '.nin_shortcut.')
            mapping.append([k0, k])

    def move_last_several_layers():
        mapping.append(['decoder.norm_out.weight', 'decoder.conv_norm_out.weight'])
        mapping.append(['decoder.norm_out.bias',   'decoder.conv_norm_out.bias'  ])
        mapping.append(['decoder.conv_out.weight', 'decoder.conv_out.weight'     ])
        mapping.append(['decoder.conv_out.bias',   'decoder.conv_out.bias'       ])
        mapping.append(['quant_conv.weight',       'quant_conv.weight'           ])
        mapping.append(['quant_conv.bias',         'quant_conv.bias'             ])
        mapping.append(['post_quant_conv.weight',  'post_quant_conv.weight'      ])
        mapping.append(['post_quant_conv.bias',    'post_quant_conv.bias'        ])

    move_first_several_layers()
    move_down_blocks()
    move_default('encoder.mid_block.resnets.0.', 'encoder.mid.block_1.')
    move_attention('encoder.mid_block.attentions.0.', 'encoder.mid.attn_1.')
    move_default('encoder.mid_block.resnets.1.', 'encoder.mid.block_2.')
    move_mid_several_layers()
    move_default('decoder.mid_block.resnets.0.', 'decoder.mid.block_1.')
    move_attention('decoder.mid_block.attentions.0.', 'decoder.mid.attn_1.')
    move_default('decoder.mid_block.resnets.1.', 'decoder.mid.block_2.')
    move_up_blocks(3, 0)
    move_up_blocks(2, 1)
    move_up_blocks(1, 2)
    move_up_blocks(0, 3)
    move_last_several_layers()

    # with open('/home/james/Project/vd-common/log/autokl-mapping.txt', 'w') as f:
    #     for ki, vi in mapping:
    #         f.write('{} | {}\n'.format(ki, vi))

    newsd = []
    for info in mapping:
        if len(info) == 2:
            to, ref = info
            newsd.append([to, sd[ref]])
        elif len(info) == 3:
            to, ref, f = info
            newsd.append([to, f(sd[ref])])
    sd = OrderedDict(newsd)
    torch.save(sd, pthout)

if operation == 'unet':
    mapping = []

    def move_tembed_blocks():
        mapping.append(['diffuser.image.time_embed.0.weight', 'time_embedding.linear_1.weight'])
        mapping.append(['diffuser.image.time_embed.0.bias',   'time_embedding.linear_1.bias'])
        mapping.append(['diffuser.image.time_embed.2.weight', 'time_embedding.linear_2.weight'])
        mapping.append(['diffuser.image.time_embed.2.bias',   'time_embedding.linear_2.bias'])

    def move_first_several_layers():
        mapping.append(['diffuser.image.data_blocks.0.0.weight', 'conv_in.weight'])
        mapping.append(['diffuser.image.data_blocks.0.0.bias',   'conv_in.bias'])

    def move_down_resblocks(ref, to):
        for k in sd.keys():
            k0 = k.replace( 'down_blocks.{}.resnets.0.'.format(ref), 'diffuser.image.data_blocks.{}.0.'.format(to))
            k0 = k0.replace('down_blocks.{}.resnets.1.'.format(ref), 'diffuser.image.data_blocks.{}.0.'.format(to+1))
            k0 = k0.replace('down_blocks.{}.downsamplers.0.conv.'.format(ref), 
                            'diffuser.image.data_blocks.{}.0.op.'.format(to+2))
            if k == k0:
                continue
            k0 = k0.replace('.norm1.weight', '.in_layers.0.weight')
            k0 = k0.replace('.norm1.bias'  , '.in_layers.0.bias')
            k0 = k0.replace('.conv1.weight', '.in_layers.2.weight')
            k0 = k0.replace('.conv1.bias'  , '.in_layers.2.bias')
            k0 = k0.replace('.time_emb_proj.', '.emb_layers.1.')
            k0 = k0.replace('.norm2.weight', '.out_layers.0.weight')
            k0 = k0.replace('.norm2.bias'  , '.out_layers.0.bias')
            k0 = k0.replace('.conv2.weight', '.out_layers.3.weight')
            k0 = k0.replace('.conv2.bias'  , '.out_layers.3.bias')
            k0 = k0.replace('.conv_shortcut', '.skip_connection')
            mapping.append([k0, k])

    def move_mid_resblocks(to):
        for k in sd.keys():
            k0 = k.replace( 'mid_block.resnets.0.', 'diffuser.image.data_blocks.{}.0.'.format(to))
            k0 = k0.replace('mid_block.resnets.1.', 'diffuser.image.data_blocks.{}.0.'.format(to+1))
            if k == k0:
                continue
            k0 = k0.replace('.norm1.weight', '.in_layers.0.weight')
            k0 = k0.replace('.norm1.bias'  , '.in_layers.0.bias')
            k0 = k0.replace('.conv1.weight', '.in_layers.2.weight')
            k0 = k0.replace('.conv1.bias'  , '.in_layers.2.bias')
            k0 = k0.replace('.time_emb_proj.', '.emb_layers.1.')
            k0 = k0.replace('.norm2.weight', '.out_layers.0.weight')
            k0 = k0.replace('.norm2.bias'  , '.out_layers.0.bias')
            k0 = k0.replace('.conv2.weight', '.out_layers.3.weight')
            k0 = k0.replace('.conv2.bias'  , '.out_layers.3.bias')
            mapping.append([k0, k])

    def move_up_resblocks(ref, to):
        for k in sd.keys():
            k0 = k.replace( 'up_blocks.{}.resnets.0.'.format(ref), 'diffuser.image.data_blocks.{}.0.'.format(to))
            k0 = k0.replace('up_blocks.{}.resnets.1.'.format(ref), 'diffuser.image.data_blocks.{}.0.'.format(to+1))
            k0 = k0.replace('up_blocks.{}.resnets.2.'.format(ref), 'diffuser.image.data_blocks.{}.0.'.format(to+2))
            k0 = k0.replace('up_blocks.{}.upsamplers.0.conv.'.format(ref), 
                            'diffuser.image.data_blocks.{}.0.conv.'.format(to+3))
            if k == k0:
                continue
            k0 = k0.replace('.norm1.weight', '.in_layers.0.weight')
            k0 = k0.replace('.norm1.bias'  , '.in_layers.0.bias')
            k0 = k0.replace('.conv1.weight', '.in_layers.2.weight')
            k0 = k0.replace('.conv1.bias'  , '.in_layers.2.bias')
            k0 = k0.replace('.time_emb_proj.', '.emb_layers.1.')
            k0 = k0.replace('.norm2.weight', '.out_layers.0.weight')
            k0 = k0.replace('.norm2.bias'  , '.out_layers.0.bias')
            k0 = k0.replace('.conv2.weight', '.out_layers.3.weight')
            k0 = k0.replace('.conv2.bias'  , '.out_layers.3.bias')
            k0 = k0.replace('.conv_shortcut', '.skip_connection')
            mapping.append([k0, k])

    def move_last_several_layers():
        mapping.append(['diffuser.image.data_blocks.29.0.0.weight', 'conv_norm_out.weight'])
        mapping.append(['diffuser.image.data_blocks.29.0.0.bias'  , 'conv_norm_out.bias'  ])
        mapping.append(['diffuser.image.data_blocks.29.0.2.weight', 'conv_out.weight'])
        mapping.append(['diffuser.image.data_blocks.29.0.2.bias'  , 'conv_out.bias'  ])

    def move_down_attn(ref, to):
        for k in sd.keys():
            k0 =  k.replace('down_blocks.{}.attentions.0.'.format(ref), 'diffuser.text.context_blocks.{}.0.'.format(to))
            k0 = k0.replace('down_blocks.{}.attentions.1.'.format(ref), 'diffuser.text.context_blocks.{}.0.'.format(to+1))
            if k == k0:
                continue
            if (k0.find('.proj_in.weight')!=-1) or (k0.find('.proj_out.weight')!=-1):
                f = lambda x: x[:, :, None, None].contiguous()
            else:
                f = lambda x: x
            mapping.append([k0, k, f])

    def move_mid_attn(to):
        for k in sd.keys():
            k0 = k.replace( 'mid_block.attentions.0.', 'diffuser.text.context_blocks.{}.0.'.format(to))
            if k == k0:
                continue
            if (k0.find('.proj_in.weight')!=-1) or (k0.find('.proj_out.weight')!=-1):
                f = lambda x: x[:, :, None, None].contiguous()
            else:
                f = lambda x: x
            mapping.append([k0, k, f])

    def move_up_attn(ref, to):
        for k in sd.keys():
            k0  = k.replace('up_blocks.{}.attentions.0.'.format(ref), 'diffuser.text.context_blocks.{}.0.'.format(to))
            k0 = k0.replace('up_blocks.{}.attentions.1.'.format(ref), 'diffuser.text.context_blocks.{}.0.'.format(to+1))
            k0 = k0.replace('up_blocks.{}.attentions.2.'.format(ref), 'diffuser.text.context_blocks.{}.0.'.format(to+2))
            if k == k0:
                continue
            if (k0.find('.proj_in.weight')!=-1) or (k0.find('.proj_out.weight')!=-1):
                f = lambda x: x[:, :, None, None].contiguous()
            else:
                f = lambda x: x
            mapping.append([k0, k, f])

    move_tembed_blocks()
    move_first_several_layers()
    move_down_resblocks(0, 1)
    move_down_resblocks(1, 4)
    move_down_resblocks(2, 7)
    move_down_resblocks(3, 10)
    move_mid_resblocks(12)
    move_up_resblocks(0, 14)
    move_up_resblocks(1, 18)
    move_up_resblocks(2, 22)
    move_up_resblocks(3, 26)
    move_last_several_layers()
    move_down_attn(0, 0)
    move_down_attn(1, 2)
    move_down_attn(2, 4)
    move_mid_attn(6)
    move_up_attn(1, 7)
    move_up_attn(2, 10)
    move_up_attn(3, 13)

    newsd = []
    for info in mapping:
        if len(info) == 2:
            to, ref = info
            newsd.append([to, sd[ref]])
        elif len(info) == 3:
            to, ref, f = info
            newsd.append([to, sd[ref]]) # SD v1.5
            # newsd.append([to, f(sd[ref])])
    sd = OrderedDict(newsd)
    torch.save(sd, pthout)


