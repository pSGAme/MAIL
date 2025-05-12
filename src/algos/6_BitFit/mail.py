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


class MAIL(nn.Module):
    def __init__(self, cfg, visual_dim=768, text_dim=512, alpha=0.0):
        super().__init__()
        self.cfg = cfg
        self.visual_a = torch.nn.Parameter(torch.ones(visual_dim))
        self.visual_b = torch.nn.Parameter(torch.zeros(visual_dim))
        # #
        self.text_a = torch.nn.Parameter(torch.ones(text_dim))
        self.text_b = torch.nn.Parameter(torch.zeros(text_dim))

        d = self.cfg.d
        visual_scale = visual_dim ** -0.5
        text_scale = text_dim ** -0.5
        t = self.cfg.t
        #
        # gaussian - 0 distribution
        self.text_proj_down = nn.Parameter(text_scale * t * torch.randn(text_dim, d))
        self.text_proj_up = nn.Parameter(visual_scale * 0 * torch.randn(d, visual_dim))

        # self.text_proj_down_bias = nn.Parameter(text_scale * t * torch.randn(text_dim, d))
        # self.text_proj_up_bias = nn.Parameter(visual_scale * 0 * torch.randn(d, visual_dim))

        # # uniform - 0 distribution
        # text_width = (12 * text_scale**2) ** 0.5
        # self.text_proj_down = nn.Parameter(text_width * torch.rand(text_dim, d) - text_width / 2)
        # self.text_proj_up = nn.Parameter(visual_scale * 0 * torch.randn(d, visual_dim))

        # # # 0 - uniform distribution
        # visual_width = (12 * visual_scale**2) ** 0.5
        # self.text_proj_down = nn.Parameter(text_scale * 0 * torch.randn(text_dim, d))
        # self.text_proj_up = nn.Parameter(visual_width * torch.rand(d, visual_dim) - visual_width / 2)

        # # # 0 - gaussian distribution
        # self.text_proj_down = nn.Parameter(text_scale * 0 * torch.randn(text_dim, d))
        # self.text_proj_up = nn.Parameter(visual_scale * torch.randn(d, visual_dim))

        # # # uniform - uniform distribution
        # text_width = (12 * text_scale ** 2) ** 0.5
        # visual_width = (12 * visual_scale**2) ** 0.5
        # self.text_proj_down = nn.Parameter(text_width * torch.rand(text_dim, d) - text_width / 2)
        # self.text_proj_up = nn.Parameter(visual_width * torch.rand(d, visual_dim) - visual_width / 2)

        # # # gaussian - gaussian distribution
        # self.text_proj_down = nn.Parameter(text_scale * torch.randn(text_dim, d))
        # self.text_proj_up = nn.Parameter(visual_scale * torch.randn(d, visual_dim))

        # 0 - 0 distribution
        # self.text_proj_down = nn.Parameter(text_scale * 0 * torch.randn(text_dim, d))
        # self.text_proj_up = nn.Parameter(visual_scale * 0 * torch.randn(d, visual_dim))


    def forward(self, x, is_text, i=0):
        if is_text:
            x = self.text_forward(x, i)
        else:
            x = self.visual_forward(x, i)
        return x

    def visual_forward(self, x, i):
        if self.cfg.start_layer <= i <= self.cfg.end_layer:
            a = self.visual_a + self.text_a @ self.text_proj_down @ self.text_proj_up
        else:
            a = self.visual_a
        b = self.visual_b
        x = x * a + b
        return x

    def text_forward(self, x, i):
        a = self.text_a
        b = self.text_b  # + self.visual_b @ self.visual_proj_down_bias  @ self.visual_proj_up_bias
        x = x * a + b
        return x


class MAIL_Linear(nn.Module):
    def __init__(self, cfg, visual_dim=768, text_dim=512, alpha=0.0):
        super().__init__()
        self.cfg = cfg
        self.visual_a = torch.nn.Parameter(torch.ones(visual_dim))
        self.visual_b = torch.nn.Parameter(torch.zeros(visual_dim))
        # #
        self.text_a = torch.nn.Parameter(torch.ones(text_dim))
        self.text_b = torch.nn.Parameter(torch.zeros(text_dim))

        d = self.cfg.d
        self.visual_scale = visual_dim ** -0.5
        self.text_scale = text_dim ** -0.5
        t = self.cfg.t
        # self.text_proj_down = nn.Parameter(text_scale * t * torch.randn(text_dim, d))
       # self.text_proj = nn.Parameter(0 * torch.randn(text_dim, visual_dim))
        self.visual_proj = nn.Parameter(0 * torch.randn(visual_dim, text_dim))

    def forward(self, x, is_text, i=0):
        if is_text:
            x = self.text_forward(x, i)
        else:
            x = self.visual_forward(x, i)
        return x

    def visual_forward(self, x, i):
        if self.cfg.start_layer <= i <= self.cfg.end_layer:
            a = self.visual_a + self.text_scale * self.text_a @ self.text_proj
        else:
            a = self.visual_a
        b = self.visual_b  #+ self.text_b  @ self.text_proj_down_bias @ self.text_proj_up_bias
        x = x * a + b
        return x

    def text_forward(self, x, i):
        a = self.text_a # + self.visual_scale * self.visual_a @ self.visual_proj
        b = self.text_b  # + self.visual_b @ self.visual_proj_down_bias  @ self.visual_proj_up_bias
        x = x * a + b
        return x

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp,  att_proj_layers_att, last_ln=None):
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


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.textNumTokens  # it should be equal to VPTNumTokens!!!!
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype  # fp32

        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.image_size
        assert cfg.maple_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"  # 9
        self.compound_prompts_depth = cfg.maple_depth  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init is not None and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype).to(device)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')


        ln_single_layer = MAIL(cfg=cfg, visual_dim=768, text_dim=512, alpha=cfg.alpha)
        # ln_single_layer = MAIL(cfg=cfg, visual_dim=1024, text_dim=768, alpha=cfg.alpha)
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
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):

        prompts = self.embedding
        return prompts, self.ln_proj_layers_mlp, self.ln_proj_layers_att, self.att_proj_layers_att, self.mlp_proj_layers_mlp
        # return prompts, None, self.compound_prompts_text, visual_deep_prompts, \
        #        self.ln_proj_layers_mlp, self.ln_proj_layers_att, self.att_proj_layers_att, self.mlp_proj_layers_mlp

class mail(nn.Module):
    def __init__(self, cfg, dict_clss, dict_doms, device):
        super().__init__()
        self.cfg = cfg
        self.dict_clss = dict_clss
        self.dict_doms = dict_doms
        self.device = device

        clip: CLIP = self.load_clip()
        self.dtype = clip.dtype

        # for text
        self.text_prompt_learner = MultiModalPromptLearner(self.cfg, self.dict_clss.keys(), clip, device)
        self.text_encoder = TextEncoder(clip)
        self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts

        self.visual_encoder = clip.visual
        # self.last_ln = None
        # self.last_adapter = None
        self.last_ln = MAIL(cfg=cfg, visual_dim=768, text_dim=512, alpha=0.0)
        self.last_adapter = MAIL(cfg=cfg, visual_dim=512, text_dim=512, alpha=0.0)

    def forward(self, image, domain_name, class_name):  # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)

        prompts, ln_proj_layers_mlp, \
        ln_proj_layers_att, mlp_proj_layers_mlp,  att_proj_layers_att = self.text_prompt_learner()  # i write dao zheli le

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts, ln_proj_layers_mlp,
                                          ln_proj_layers_att, mlp_proj_layers_mlp,  att_proj_layers_att, self.last_ln)
        image_features = self.visual_encoder(image.type(self.dtype),
                                             ln_proj_layers_mlp,
                                             ln_proj_layers_att, mlp_proj_layers_mlp,  att_proj_layers_att, self.last_ln)
        if self.last_adapter:
            image_features = self.last_adapter(image_features, is_text=False, i=12)
            text_features = self.last_adapter(text_features, is_text=True, i=12)

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

        trainer_name = 'MaPLe' if self.cfg.maple_depth > 0 else 'CoOp'
        design_details = {"trainer": trainer_name,
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0,
                          "maple_length": self.cfg.maple_length}

        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model.float().to(self.device)
