import sys
import os
code_path = '/home' # e.g. '/home/username/ProS'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))
import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP, VisionTransformer
import math
import torch.nn as nn
from PIL import Image
from functools import reduce
from operator import mul
from src.utils import utils
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model:CLIP):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype) # 300, 77, 512
        x = x.permute(1, 0, 2)  # NLD -> LND
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
        n_cls = len(classnames) # 400
        n_ctx = cfg.tp_N_CTX  # number of context tokens 16
        dtype = clip_model.dtype # float32
        ctx_dim = clip_model.ln_final.weight.shape[0] # LayerNorm第0维，前一层的输出维度 512
        clip_imsize = clip_model.visual.input_resolution # 输入图片大小 # 224
        cfg_imsize = cfg.image_size # 设定的输入图片大小
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device) # 16, 512
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx) # 生成n_ctx个 X，eg. X X X X X

        print(f'Initial context: "{prompt_prefix}"') # 'X X X X X X X X X X X X X X X X'
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized 16,512

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] # 400，类名的长度
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # xxxxxxxxx classname .

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device) # 将prompt中的句子的每个单词转换成字典中的数字，长度固定为77，多的用0补齐, [400,77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # 400, 77, 512

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # 400,16,512

        prefix = self.token_prefix # 400,1,512
        suffix = self.token_suffix # 400,60,512


        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts # 400,77,512

class DomainPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model:CLIP, device):
        super().__init__()
        n_cls = len(classnames) # 300
        n_ctx = cfg.tp_N_CTX  # number of context tokens 16
        dtype = clip_model.dtype # float32
        self.ctx_dim = clip_model.ln_final.weight.shape[0] # LayerNorm第0维，前一层的输出维度 512
        clip_imsize = clip_model.visual.input_resolution # 输入图片大小 # 224
        cfg_imsize = cfg.image_size # 设定的输入图片大小
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        # print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=dtype).to(device) # 16, 512
        nn.init.normal_(ctx_vectors, std=0.02)
        # prompt_prefix = " ".join(["X"] * n_ctx) # 生成n_ctx个 X，eg. X X X X X

        # print(f'Initial context: "{prompt_prefix}"') # 'X X X X X X X X X X X X X X X X'
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized 16,512 

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] # 300，类名的长度, 有的長度是2，比如aircraft carrier
        prompts = ["a photo of " + name + " from " + "X "*n_ctx +"domain." for name in classnames] # xxxxxxxxx classname .
        self.prefix_index = [length+5 for length in name_lens] # SOS a photo of classname from 
        print("Text Prompt Example:" + prompts[0])
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device) # 将prompt中的句子的每个单词转换成字典中的数字，长度固定为77，多的用0补齐, [300,77]

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # 300, 77, 512
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        self.register_buffer("origin_text_embedding",embedding)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor


    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # 300,16,512
        
        prompts = [torch.cat([self.origin_text_embedding[i,:self.prefix_index[i]],ctx[i],self.origin_text_embedding[i,self.prefix_index[i]+self.n_ctx:]],dim=0).view(1,-1,self.ctx_dim) for i in range(self.n_cls)]
        prompts = torch.cat(prompts, dim=0)
        return prompts # 300,77,512


class PromptedViT(nn.Module):
    def __init__(self, config, model: CLIP):
        super(PromptedViT, self).__init__()
        self.config = config

        self.conv1 = model.visual.conv1
        width = self.conv1.out_channels
        self.class_embedding = model.visual.class_embedding #  self.feature_template = clip.visual.class_embedding
        self.positional_embedding = model.visual.positional_embedding #  self.clip_positional_embedding = clip.visual.positional_embedding  # 768
        self.ln_pre = model.visual.ln_pre
        self.transformer = model.visual.transformer
        self.ln_post = model.visual.ln_post
        self.proj = model.visual.proj #  self.feature_proj = clip.visual.proj

        patch_size = self.conv1.kernel_size
        self.num_tokens = self.config.vp_NUM_TOKENS  # "10"  # number of prompted tokens

        if self.config.vp_PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.config.vp_PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        if self.config.vp_INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_tokens, prompt_dim))  # layer, num_token, prompt_dim
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            # scale = width ** -0.5
            # self.ge_cls_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_CLS_NUM_TOKENS, width))
            # self.ge_dom_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_DOM_NUM_TOKENS, width))
            # which is better?

            if self.config.vp_DEEP:  # noqa
                total_d_layer = self.transformer.layers - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, self.num_tokens, prompt_dim))
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
                # also, which is better?
        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]  # batch size
        # after CLS token, all before image patches
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 65 768 49
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)  # [65, 50, 678] + [50 ,768]

        x = torch.cat((
            x[:, :1, :],  # CLS token
            self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1),
            x[:, 1:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x

    def forward_deep_prompt(self, embedding_output):
        hidden_states = None
        B = embedding_output.shape[0]
        num_layers = self.transformer.layers
        # print("yes")
        for i in range(num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_proj(self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1)

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1 + self.num_tokens):, :]
                    ), dim=1)

                hidden_states = self.transformer.resblocks[i](hidden_states)
        return hidden_states

    def forward(self, x):
        # this is the default version:
        x = self.incorporate_prompt(x)

        if self.config.vp_DEEP:
            x = self.ln_pre(x)  # should exist?
            x = x.permute(1, 0, 2)
            x = self.forward_deep_prompt(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
        else:
            x = self.ln_pre(x) # should exist?
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
            # x = x.permute(1, 0, 2)
            # x = self.transformer(x)
            # x = x.permute(1, 0, 2)
            # x = self.ln_post(x[:, out_token, :])
            # x = x @ self.feature_proj
        return x


class prosnet(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        clip:CLIP = self.load_clip()

        # for text
        self.text_encoder = None
        self.tokenized_prompts = None

        if self.cfg.text == 'None':
            self.text_encoder = clip.encode_text
        else:
            if self.cfg.text == "CoOp":
                self.text_prompt_learner = PromptLearner(self.cfg, self.dict_clss.keys(), clip, device)
            else: # self.cfg.text == "DCoOp"
                self.text_prompt_learner = DomainPromptLearner(self.cfg, self.dict_clss.keys(), clip,
                                                               device)  # look here
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts

        self.visual_encoder = PromptedViT(self.cfg, clip)

        # # for visual
        # self.ge_cls_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_CLS_NUM_TOKENS, width))
        # self.ge_dom_prompt_template = nn.Parameter(scale * torch.randn(self.cfg.GP_DOM_NUM_TOKENS, width))


    def incorporate_prompt(self, x):
        B = x.shape[0]
        x = self.conv1(x)
        # print(x.shape) # 400, 768, 7, 7
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 400 768 49
        x = x.permute(0, 2, 1)  # 400 49 768
        x = torch.cat((
            (self.feature_template + self.clip_positional_embedding[0]).expand(B, -1).view(B, 1, -1),
            self.ge_dom_prompt_template.expand(B, -1, -1),
            self.ge_cls_prompt_template.expand(B, -1, -1),
            x + self.clip_positional_embedding[1:]
        ), dim=1)  # print(x.shape) # 400, 52, 768
        return x
    
    def vit(self, x, out_token):
        # 少了 ln_pre :)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:,out_token,:])
        x = x @ self.feature_proj

        return x

    def image_encoder(self, image):
        x = self.incorporate_prompt(image)
        x = self.vit(x, 1)
        return x


    def forward(self, image, class_name): # bt 3, 244, 244
        image_features = self.image_encoder(image) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else :
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)

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