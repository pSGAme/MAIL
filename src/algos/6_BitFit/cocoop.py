import sys
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from clip import clip
from clip import CLIP
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
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

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        #print(x.shape)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.textNumTokens
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize =  cfg.image_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
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
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        # if cfg.TRAINER.COCOOP.PREC == "fp16":
        #     self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class maple(nn.Module):
    def __init__(self, cfg, dict_clss, dict_doms, device):
        super().__init__()
        self.cfg = cfg
        self.dict_clss = dict_clss
        self.dict_doms = dict_doms
        self.device = device

        clip: CLIP = self.load_clip()
        self.dtype = clip.dtype

        # for text
        self.text_prompt_learner = PromptLearner(self.cfg, self.dict_clss.keys(), clip, device)

        self.text_encoder = TextEncoder(clip)
        self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts

        self.visual_encoder = clip.visual

        # self.visual_encoder = PromptedViT(self.cfg, clip)

    def forward(self, image, domain_name=0, class_name=0):  # bt 3, 244, 244
        #cls_id = utils.numeric_classes(class_name, self.dict_clss)
        #dom_id = utils.numeric_classes(domain_name, self.dict_doms)
        tokenized_prompts = self.tokenized_prompts
        # prompts, text_adapter_func, visual_adapter_func = self.text_prompt_learner()
        #

        #
        # text_features = self.text_encoder(
        #     prompts, tokenized_prompts, text_adapter_func
        # )
        # image_features = self.visual_encoder([image, visual_adapter_func])
        #
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.visual_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #print(f"memory before:{torch.cuda.memory_allocated() / (1024 ** 3)} GBs")
        prompts = self.text_prompt_learner(image_features)
       # print(f"memory after:{torch.cuda.memory_allocated() / (1024 ** 3)} GBs")

        text_features_list = []
        for pts_i in prompts:
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_list.append(text_features)
        return image_features, text_features_list

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
