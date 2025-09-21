import sys
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from clip_mma import clip
from clip_mma import CLIP
from clip_mma.simple_tokenizer import SimpleTokenizer as _Tokenizer
import math
import torch.nn as nn
from PIL import Image
from functools import reduce
from operator import mul
from src.utils import utils
import copy

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
_tokenizer = _Tokenizer()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, return_adapater_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if return_adapater_func == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, return_adapater_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x



class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()

        self.n_cls = len(classnames)

        self.ctx_init = cfg.ctx_init

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.image_size

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self._build_text_embedding(device, classnames, clip_model)

        # build multi-modal adapter
        self.text_adapter_func = lambda x: self.return_text_adapter(index=x)
        self.text_adapter = self._build_adapter(
            clip_model.ln_final.weight.shape[0],
            len(clip_model.transformer.resblocks),
            1,
            12,
            32,
            clip_model.dtype
        )

        self.visual_adapter_func = lambda x: self.return_visual_adapter(index=x)
        self.visual_adapter = self._build_adapter(
            clip_model.visual.ln_post.weight.shape[0],
            len(clip_model.visual.transformer.resblocks),
            1,
            12,
            32,
            clip_model.dtype
        )

        self.shared_adapter = nn.Identity()

        self.adapter_scale = 0.001

    def return_text_adapter(self, index):
        return self.text_adapter[index], self.shared_adapter, self.adapter_scale

        #return None, self.shared_adapter, self.adapter_scale

    # self.text_adapter[index]
    def return_visual_adapter(self, index):
        return self.visual_adapter[index], self.shared_adapter, self.adapter_scale

    def _build_text_embedding(self, device, classnames, clip_model):
        dtype = clip_model.dtype
        text_ctx_init = 'a photo of a'

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [text_ctx_init + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(
            device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def _build_adapter(self, d_model, n_layers, l_start, l_end, mid_dim, dtype):

        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end + 1):
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        return adapter

    def forward(self):
        embedding = self.token_embedding

        if self.text_adapter[0] is not None:
            token_embedding = self.text_adapter[0].down(embedding)
            shared_adapter = self.shared_adapter[0]
            token_embedding = shared_adapter(token_embedding)
            token_embedding = self.text_adapter[0].up(token_embedding)
            embedding = embedding + self.adapter_scale * token_embedding
        return embedding, self.text_adapter_func, self.visual_adapter_func


class mma(nn.Module):
    def __init__(self, cfg, dict_clss, dict_doms, device):
        super().__init__()
        self.cfg = cfg
        self.dict_clss = dict_clss
        self.dict_doms = dict_doms
        self.device = device

        clip: CLIP = self.load_clip()
        self.dtype = clip.dtype

        # for text
        self.text_prompt_learner = AdapterLearner(self.cfg, self.dict_clss.keys(), clip, device)

        self.text_encoder = TextEncoder(clip)
        self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts

        self.visual_encoder = clip.visual

        # self.visual_encoder = PromptedViT(self.cfg, clip)

    def forward(self, image, domain_name, class_name):  # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)

        prompts, text_adapter_func, visual_adapter_func = self.text_prompt_learner()

        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(
            prompts, tokenized_prompts, text_adapter_func
        )
        image_features = self.visual_encoder([image, visual_adapter_func])

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features

    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)
