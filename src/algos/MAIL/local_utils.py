import torchvision.transforms as transforms
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from src.data.DomainNet import domainnet
from src.data.Sketchy import sketchy_extended
from src.data.TUBerlin import tuberlin_extended

# testing packages
import torch
from tqdm import tqdm
from src.utils import utils, GPUmanager
from src.utils.metrics import compute_retrieval_metrics
gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

im_mean = [0.485, 0.456, 0.406]
im_std = [0.229, 0.224, 0.225]

# Image transformations
def get_transform(image_size):
    image_transforms = {
        'train':
            transforms.Compose([
                transforms.RandomResizedCrop((image_size, image_size), (0.8, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.ToTensor(),
                transforms.Normalize(im_mean, im_std)
            ]),
        'eval': transforms.Compose([
            transforms.Resize(image_size, interpolation=BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    }
    return image_transforms


def get_data_info(args, dataset_name):
    data_input = {}
    if dataset_name == 'Sketchy':
        data_input = sketchy_extended.create_trvalte_splits(args)
    if dataset_name == 'DomainNet':
        data_input = domainnet.create_trvalte_splits(args)
    if dataset_name == 'TUBerlin':
        data_input = tuberlin_extended.create_trvalte_splits(args)

    tr_classes = data_input['tr_classes']
    # va_classes = data_input['va_classes']
    te_classes = data_input['te_classes']
    data_splits = data_input['splits']
    return tr_classes, te_classes, data_splits


@torch.no_grad()
def evaluate(loader_sketch, loader_image, model, dict_clss, args):
    model.eval()
    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in tqdm(enumerate(loader_sketch), desc='Extrac query feature', total=len(loader_sketch)):
        sk = sk.float().to(device)
        cls_id = utils.numeric_classes(cls_sk, dict_clss)
        prompts, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att = model.mail_learner()
        sk_em = model.visual_encoder(sk, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp,
                                     att_proj_layers_att, model.last_ln)
        if model.last_adapter is not None:
            sk_em = model.last_adapter(sk_em, is_text=False, i=1)
        sketchEmbeddings.append(sk_em)
        cls_numeric = torch.from_numpy(cls_id).long().to(device)
        sketchLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in tqdm(enumerate(loader_image), desc='Extrac gallery feature', total=len(loader_image)):
        im = im.float().to(device)
        cls_id = utils.numeric_classes(cls_im, dict_clss)
        prompts, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att = model.mail_learner()
        im_em = model.visual_encoder(im, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp,
                                     att_proj_layers_att, model.last_ln)
        if model.last_adapter is not None:
            im_em = model.last_adapter(im_em, is_text=False, i=1)
        realEmbeddings.append(im_em)
        cls_numeric = torch.from_numpy(cls_id).long().to(device)
        realLabels.append(cls_numeric)

    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('Query Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    print("Computing un-normed situation:")
    eval_data_unnorm = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

    print("Computing normed situation:")
    sketchEmbeddings = sketchEmbeddings / sketchEmbeddings.norm(dim=-1, keepdim=True)
    realEmbeddings = realEmbeddings / realEmbeddings.norm(dim=-1, keepdim=True)
    eval_data_norm = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)
    return eval_data_unnorm, eval_data_norm


def get_folder_name(args):
    if args.dataset == 'DomainNet':
        save_folder_name = 'seen-' + args.seen_domain + '_unseen-' + args.holdout_domain + '_x_' + args.gallery_domain
        if not args.include_auxillary_domains:
            save_folder_name += '_noaux'
    elif args.dataset == 'Sketchy':
        if args.is_eccv_split:
            save_folder_name = 'eccv_split'
        else:
            save_folder_name = 'random_split'
    else:
        save_folder_name = ''
    return save_folder_name