import torch
import torch.nn as nn
import numpy as np
from functools import partial
from lib.model_zoo.common.get_model import register

version = '0'
symbol = 'clip'

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

from transformers import CLIPTokenizer, CLIPTextModel

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

@register('clip_text_frozen', version)
class FrozenCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

from transformers import CLIPProcessor, CLIPVisionModel

@register('clip_vision_frozen', version)
class FrozenCLIPVisionEmbedder(AbstractEncoder):
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(version)
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs['pixel_values'].to(self.device)
        outputs = self.transformer(pixel_values=pixels)
        z = outputs.last_hidden_state
        return z

    def encode(self, image):
        return self(image)

from transformers import CLIPModel

@register('clip_frozen', version)
class FrozenCLIP(AbstractEncoder):
    def __init__(self, 
                 version="openai/clip-vit-large-patch14", 
                 max_length=77, 
                 encode_type='encode_text',):  # clip-vit-base-patch32
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        self.model = CLIPModel.from_pretrained(version)
        self.max_length = max_length  # TODO: typical value?
        self.encode_type = encode_type
        self.pinv_text_projection = None
        self.freeze()

    def get_device(self):
        # A trick to get device
        return self.model.text_projection.weight.device

    def freeze(self):
        self.model = self.model.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def encode_text_pooled(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        return self.model.get_text_features(input_ids=tokens)

    def encode_vision_pooled(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs['pixel_values'].to(self.get_device())
        return self.model.get_image_features(pixel_values=pixels)

    def encode_text_noproj(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        outputs = self.model.text_model(input_ids=tokens)
        return outputs.last_hidden_state
        
    def encode_vision_noproj(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs['pixel_values'].to(self.get_device())
        outputs = self.model.vision_model(pixel_values=pixels)
        return outputs.last_hidden_state

    def encode_text_bug(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        outputs = self.model.text_model(input_ids=tokens)
        z = outputs.last_hidden_state
        z_pooled = outputs.pooler_output
        z = z / torch.norm(z_pooled.unsqueeze(1), dim=-1, keepdim=True)
        return self.model.text_projection(z)

    def encode_text(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        outputs = self.model.text_model(input_ids=tokens)
        z = self.model.text_projection(outputs.last_hidden_state)
        z_pooled = self.model.text_projection(outputs.pooler_output)
        z = z / torch.norm(z_pooled.unsqueeze(1), dim=-1, keepdim=True)
        return z

    def encode_vision(self, images):
        z = self.encode_vision_noproj(images)
        z = self.model.vision_model.post_layernorm(z)
        z = self.model.visual_projection(z)
        z_pooled = z[:, 0:1]
        # z_pooled_normed = z_pooled / z_pooled.norm(dim=-1, keepdim=True)
        z = z / torch.norm(z_pooled, dim=-1, keepdim=True)
        return z

    def encode_vision_pinvtext(self, images):
        blank_text_encode_norm_avg = 28.9096
        z = self.encode_vision(images)
        if self.pinv_text_projection is None:
            self.pinv_text_projection = torch.linalg.pinv(self.model.text_projection.weight).T
        z = torch.matmul(z, self.pinv_text_projection)
        # z = z / torch.norm(z[:, 0:1], dim=-1, keepdim=True)
        z = z / torch.norm(z, dim=-1, keepdim=True)
        z = z*blank_text_encode_norm_avg
        # return z[:, 1:2].repeat(1, 77, 1)
        z2 = self.encode_text_noproj('')
        # z2[:, 1:77] = z[:, 0:76]
        return torch.flip(z, dims=(1,))[:, 0:77]

    def encode(self, *args, **kwargs):
        return getattr(self, self.encode_type)(*args, **kwargs)

#############################
# copyed from justin's code #
#############################

@register('clip_vision_frozen_justin', version)
class FrozenCLIPVisionEmbedder_Justin(AbstractEncoder):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model='ViT-L/14',
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        from . import clip_justin
        self.model, _ = clip_justin.load(name=model, device=device, jit=jit)
        self.device = device
        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

        # I didn't call this originally, but seems like it was frozen anyway
        self.freeze()

    def freeze(self):
        self.transformer = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        import kornia
        # Expects inputs in the range -1, 1
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x)).float()

    def encode(self, im):
        return self(im).unsqueeze(1)
