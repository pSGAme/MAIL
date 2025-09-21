import sys
import os

import torch
import torch.nn as nn
from clip_mail import clip
from clip_mail.model import CLIP
from clip_mail.simple_tokenizer import SimpleTokenizer as _Tokenizer
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


# just do nothing
class BitFit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, is_text, i=0):
        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp,
                att_proj_layers_att, last_ln=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, 0, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att]
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        if last_ln is not None:
            x = last_ln(self.ln_final(x).type(self.dtype), is_text=True, i=12)
        else:
            x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class BitFitLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype  # fp32

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.image_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        prompt_prefix = ctx_init.replace("_", " ")  # "a photo of [CLASS]."
        print(f'Initial context: "{prompt_prefix}"')

        ln_single_layer = BitFit()
        self.ln_proj_layers_mlp = _get_clones(ln_single_layer, cfg.ivlu_end_layer)
        self.ln_proj_layers_att = _get_clones(ln_single_layer, cfg.ivlu_end_layer)
        self.att_proj_layers_att = _get_clones(ln_single_layer, cfg.ivlu_end_layer)
        self.mlp_proj_layers_mlp = _get_clones(ln_single_layer, cfg.ivlu_end_layer)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)

        self.register_buffer("embedding", embedding)  # CLS, EOS

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        prompts = self.embedding
        return prompts, self.ln_proj_layers_mlp, self.ln_proj_layers_att, self.att_proj_layers_att, self.mlp_proj_layers_mlp


class bitfit(nn.Module):
    def __init__(self, cfg, dict_clss, dict_doms, device):
        super().__init__()
        self.cfg = cfg
        self.dict_clss = dict_clss
        self.dict_doms = dict_doms
        self.device = device

        clip: CLIP = self.load_clip()
        self.dtype = clip.dtype

        self.text_prompt_learner = BitFitLearner(self.cfg, self.dict_clss.keys(), clip, device)
        self.text_encoder = TextEncoder(clip)
        self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        self.visual_encoder = clip.visual
        self.last_ln = None

    def forward(self, image, domain_name, class_name):  # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)

        prompts, ln_proj_layers_mlp, \
        ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att = self.text_prompt_learner()

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts, ln_proj_layers_mlp,
                                          ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att, self.last_ln)
        image_features = self.visual_encoder(image.type(self.dtype),
                                             ln_proj_layers_mlp,
                                             ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att, self.last_ln)

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
