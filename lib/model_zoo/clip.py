import torch
import torch.nn as nn
import numpy as np
from functools import partial
from lib.model_zoo.common.get_model import register
import torch.nn.functional as F

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

###############
# for vd next #
###############

from transformers import CLIPModel

@register('clip_text_context_encoder')
class CLIPTextContextEncoder(AbstractEncoder):
    def __init__(self, 
                 version="openai/clip-vit-large-patch14", 
                 max_length=77, 
                 fp16=False, ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.model = CLIPModel.from_pretrained(version)
        self.max_length = max_length
        self.fp16 = fp16
        self.freeze()

    def get_device(self):
        # A trick to get device
        return self.model.text_projection.weight.device

    def freeze(self):
        self.model = self.model.eval()
        self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False
        
    def encode(self, text):
        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        outputs = self.model.text_model(input_ids=tokens)
        z = self.model.text_projection(outputs.last_hidden_state)
        z_pooled = self.model.text_projection(outputs.pooler_output)
        z = z / torch.norm(z_pooled.unsqueeze(1), dim=-1, keepdim=True)
        return z

from transformers import CLIPProcessor

@register('clip_image_context_encoder')
class CLIPImageContextEncoder(AbstractEncoder):
    def __init__(self, 
                 version="openai/clip-vit-large-patch14", 
                 fp16=False, ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        self.model = CLIPModel.from_pretrained(version)
        self.fp16 = fp16
        self.freeze()

    def get_device(self):
        # A trick to get device
        return self.model.text_projection.weight.device

    def freeze(self):
        self.model = self.model.eval()
        self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def _encode(self, images):
        if isinstance(images, torch.Tensor):
            import torchvision.transforms as tvtrans
            images = [tvtrans.ToPILImage()(i) for i in images]
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs['pixel_values'].half() if self.fp16 else inputs['pixel_values']
        pixels = pixels.to(self.get_device())
        outputs = self.model.vision_model(pixel_values=pixels)
        z = outputs.last_hidden_state
        z = self.model.vision_model.post_layernorm(z)
        z = self.model.visual_projection(z)
        z_pooled = z[:, 0:1]
        z = z / torch.norm(z_pooled, dim=-1, keepdim=True)
        return z

    @torch.no_grad()
    def _encode_wmask(self, images, masks):
        assert isinstance(masks, torch.Tensor)
        assert (len(masks.shape)==4) and (masks.shape[1]==1)
        masks = torch.clamp(masks, 0, 1)
        masks = masks.float()
        masks = F.interpolate(masks, [224, 224], mode='bilinear')
        if masks.sum() == masks.numel():
            return self._encode(images)

        device = images.device
        dtype = images.dtype
        gscale = masks.mean(axis=[1, 2, 3], keepdim=True).flatten(2)

        vtoken_kernel_size = self.model.vision_model.embeddings.patch_embedding.kernel_size
        vtoken_stride = self.model.vision_model.embeddings.patch_embedding.stride
        mask_kernal = torch.ones([1, 1, *vtoken_kernel_size], device=device, requires_grad=False).float()
        vtoken_mask = torch.nn.functional.conv2d(masks, mask_kernal, stride=vtoken_stride).flatten(2).transpose(1, 2)
        vtoken_mask = vtoken_mask/np.prod(vtoken_kernel_size)
        vtoken_mask = torch.concat([gscale, vtoken_mask], axis=1)

        import types
        def customized_embedding_forward(self, pixel_values):
            batch_size = pixel_values.shape[0]
            patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
            embeddings = embeddings + self.position_embedding(self.position_ids)
            embeddings = embeddings*vtoken_mask.to(embeddings.dtype)
            return embeddings

        old_forward = self.model.vision_model.embeddings.forward
        self.model.vision_model.embeddings.forward = types.MethodType(
            customized_embedding_forward, self.model.vision_model.embeddings)

        z = self._encode(images)
        self.model.vision_model.embeddings.forward = old_forward
        z = z * vtoken_mask.to(dtype)
        return z

    def encode(self, images, masks=None):
        if masks is None:
            return self._encode(images)
        else:
            return self._encode_wmask(images, masks)
