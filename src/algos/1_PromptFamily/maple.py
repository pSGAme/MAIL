
import torch
from clip_maple import clip
from clip_maple.model import CLIP
from clip_maple.simple_tokenizer import SimpleTokenizer as _Tokenizer
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

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass

        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class FixedEmbeddings():
    def __init__(self, cfg, classnames, clip_model):
        self.clip_model = clip_model
        clip_imsize = clip_model.visual.input_resolution  # 输入图片大小 # 224
        cfg_imsize = cfg.image_size  # 设定的输入图片大小
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        prompt_prefix = "a photo of a"
        print('Vision Prompting Design')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Using fixed hand crated prompts")

        # print(next(clip_model.parameters()).device)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

    def return_fixed_embeddings(self):
        text_features = self.clip_model.encode_text(self.tokenized_prompts)
        return text_features


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)  # 400
        n_ctx = cfg.textNumTokens  # number of context tokens 16
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype  # float32
        ctx_dim = clip_model.ln_final.weight.shape[0]  # LayerNorm第0维，前一层的输出维度 512
        clip_imsize = clip_model.visual.input_resolution  # 输入图片大小 # 224
        cfg_imsize = cfg.image_size  # 设定的输入图片大小
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
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device)  # 16, 512
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)  # 生成n_ctx个 X，eg. X X X X X

        print(f'Initial context: "{prompt_prefix}"')  # 'X X X X X X X X X X X X X X X X'
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized 16,512

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]  # 400，类名的长度
        prompts = [prompt_prefix + " " + name + "." for name in classnames]  # xxxxxxxxx classname .

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(
            device)  # 将prompt中的句子的每个单词转换成字典中的数字，长度固定为77，多的用0补齐, [400,77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # 400, 77, 512

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # 400,16,512

        prefix = self.token_prefix  # 400,1,512
        suffix = self.token_suffix  # 400,60,512

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts  # 400,77,512


class NonePromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()

    def forward(self):
        return None, None, None, None


class IVLPPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)
        self.visual_len = cfg.visualNumTokens
        self.visual_depth = cfg.visualDepth
        self.text_len = cfg.textNumTokens  # it should be equal to VPTNumTokens!!!!
        self.text_depth = cfg.textDepth
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype  # fp32

        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.image_size

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        if self.text_len > 0:
            if ctx_init is not None and self.text_len <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                prompt = clip.tokenize(ctx_init).to(device)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype).to(device)
                ctx_vectors = embedding[0, 1: 1 + self.text_len, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(self.text_len, ctx_dim, dtype=dtype).to(device)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * self.text_len)
            self.ctx = nn.Parameter(ctx_vectors)
            self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_len, 512))
                                                           for _ in range(self.text_depth - 1)])
            for single_para in self.compound_prompts_text:
                nn.init.normal_(single_para, std=0.02)
        else:
            prompt_prefix = ctx_init.replace("_", " ")
        print('Text Design:')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.text_len}")
        print(f"Depth of context words (tokens): {self.text_depth}")

        print('Image Design:')
        print(f"Number of context words (tokens): {self.visual_len}")
        print(f"Depth of context words (tokens): {self.visual_depth}")

        if self.visual_len > 0:
            conv1 = clip_model.visual.conv1
            width = conv1.out_channels
            patch_size = conv1.kernel_size
            prompt_dim = width
            scale = width ** -0.5
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            # self.compound_prompts_visual = nn.ParameterList([nn.Parameter(torch.empty(1, self.visual_len, 768))
            #                                                for _ in range(self.visual_depth)])
            # for single_para in self.compound_prompts_visual:
            #     nn.init.normal_(single_para.data, -val, val)

            self.compound_prompts_visual = nn.ParameterList(
                [nn.Parameter(scale * torch.randn(1, self.visual_len, width)) for _ in range(self.visual_depth)])

        # self.prompt_embeddings= nn.Parameter(scale * torch.randn(1, self.num_tokens, width))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.text_len:, :])  # CLS, EOS
        self.register_buffer("fixed_prompts", embedding)  # CLS, EOS

        self.n_cls = n_cls
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

    def forward(self):

        if self.text_len > 0 and self.training:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
            compound_prompts_text = self.compound_prompts_text
        else:
            prompts = self.fixed_prompts
            compound_prompts_text = []

        if self.visual_len > 0:
            visual_deep_prompts = self.compound_prompts_visual[1:]
            visual_prompts = self.compound_prompts_visual[0]
        else:
            visual_prompts = None
            visual_deep_prompts = []
        return prompts, visual_prompts, compound_prompts_text, visual_deep_prompts


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
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)

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

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768

        return prompts, self.proj(
            self.ctx), self.compound_prompts_text, visual_deep_prompts  # pass here original, as for visual 768 is required
        # the first two corresponding the first two layers
        # while the second corresponding the latter two layers


class maple(nn.Module):
    def __init__(self, cfg, dict_clss, dict_doms, device):
        super().__init__()
        self.cfg = cfg
        self.dict_clss = dict_clss
        self.dict_doms = dict_doms
        self.device = device

        clip: CLIP = self.load_clip()
        self.dtype = clip.dtype
        print(self.dtype)

        # for text
        if cfg.maple:
            self.text_prompt_learner = MultiModalPromptLearner(self.cfg, self.dict_clss.keys(), clip, device)
        else:
            self.text_prompt_learner = IVLPPromptLearner(self.cfg, self.dict_clss.keys(), clip, device)
        self.text_encoder = TextEncoder(clip)
        self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts

        self.visual_encoder = clip.visual

    def forward(self, image, domain_name, class_name):  # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.text_prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.visual_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

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

        maple_length = self.cfg.maple_length
        if not self.cfg.maple:
            maple_length = max(self.cfg.textNumTokens, self.cfg.visualNumTokens)
        design_details = {"maple_length": maple_length,
                          "maple": self.cfg.maple,
                          "text_length": self.cfg.textNumTokens,
                          "image_length": self.cfg.visualNumTokens}

        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model.float().to(self.device)
