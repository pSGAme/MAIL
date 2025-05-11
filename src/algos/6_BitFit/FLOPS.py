import sys
from tqdm import tqdm
import os
# code_path = '/home/user/Code/DePro' # e.g. '/home/username/ProS'
# sys.path.append(code_path)
# sys.path.append(os.path.join(code_path, "src"))
# sys.path.append(os.path.join(code_path, "clip"))

from cocoop import maple
import torch
import math
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.DomainNet import domainnet
from src.data.Sketchy import sketchy_extended
from src.data.TUBerlin import tuberlin_extended
import numpy as np
import torch.backends.cudnn as cudnn
from src.data.dataloaders import CuMixloader, BaselineDataset
from src.data.sampler import BalancedSampler, MoreBalancedSampler, FewShotSampler
from src.utils import utils, GPUmanager
from src.utils.logger import AverageMeter
from src.utils.metrics import compute_retrieval_metrics
from PIL import Image
from src.losses.sup_con_loss import soft_sup_con_loss
from src.losses.sup_con_loss import triplet_loss
from torch import optim
from src.utils.logging import Logger

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
gm = GPUmanager.GPUManager()
# gpu_index = gm.auto_choice()
#
# device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

device = "cuda:0"


class Trainer:

    def __init__(self, args):
        self.args = args
        print('\nLoading data...')
        if args.dataset == 'Sketchy':
            data_input = sketchy_extended.create_trvalte_splits(args)
        if args.dataset == 'DomainNet':
            data_input = domainnet.create_trvalte_splits(args)
        if args.dataset == 'TUBerlin':
            data_input = tuberlin_extended.create_trvalte_splits(args)

        self.tr_classes = data_input['tr_classes']
        self.va_classes = data_input['va_classes']
        self.te_classes = data_input['te_classes']
        self.data_splits = data_input['splits']
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)

        # Imagenet standards
        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        # Image transformations
        self.image_transforms = {
            'train':
                transforms.Compose([
                    transforms.RandomResizedCrop((args.image_size, args.image_size), (0.8, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(im_mean, im_std)
                ]),
            'eval': transforms.Compose([
                transforms.Resize(args.image_size, interpolation=BICUBIC),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                # lambda image: image.convert("RGB"),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        }

        # class dictionary
        self.dict_clss = utils.create_dict_texts(self.tr_classes)  # 生成 类:index 的一个字典

        # print(self.dict_clss). word to one-hot, not vector
        self.te_dict_class = utils.create_dict_texts(self.tr_classes + self.va_classes + self.te_classes)
        # print(self.te_dict_class)

        # self.te_dict_class = utils.create_dict_texts(self.tr_classes+ self.te_classes)
        # print(self.te_dict_class)
        fls_tr = self.data_splits['tr']
        cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
        dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
        tr_domains_unique = np.unique(dom_tr)

        # doamin dictionary
        self.dict_doms = utils.create_dict_texts(tr_domains_unique)

        domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)

        data_train = CuMixloader(fls_tr, cls_tr, dom_tr, self.dict_doms, transforms=self.image_transforms['train'])

        cls_dis = utils.numeric_classes(cls_tr, self.dict_clss)

        # train_sampler = MoreBalancedSampler(domain_ids, cls_dis,
        #                                     domains_per_batch=len(tr_domains_unique))  # 每个batch的采样都来自同一个domain
        train_sampler = FewShotSampler(domain_ids, cls_dis,
                                          domains_per_batch=len(tr_domains_unique))  # 每个batch的采样都来自同一个domain
        # train_sampler = BalancedSampler(domain_ids, args.batch_size // len(tr_domains_unique),
        #                                 domains_per_batch=len(tr_domains_unique))  # 每个batch的采样都来自同一个domain

        self.train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)
        self.train_loader_for_SP = DataLoader(dataset=data_train, batch_size=400, sampler=train_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
        data_va_query = BaselineDataset(self.data_splits['query_va'], transforms=self.image_transforms['eval'])
        data_va_gallery = BaselineDataset(self.data_splits['gallery_va'], transforms=self.image_transforms['eval'])
        # data_va_query = BaselineDataset(data_splits['query_va'],)
        # data_va_gallery = BaselineDataset(data_splits['gallery_va'])

        # PyTorch valid loader for query
        self.va_loader_query = DataLoader(dataset=data_va_query, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True)
        # PyTorch valid loader for gallery
        self.va_loader_gallery = DataLoader(dataset=data_va_gallery, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

        print(
            f'#Tr samples:{len(data_train)}; #Val queries:{len(data_va_query)}; #Val gallery samples:{len(data_va_gallery)}.\n')
        print('Loading Done\n')

        self.model = maple(self.args, self.dict_clss, self.dict_doms, device)
        self.model = self.model.to(device)

        if args.dataset == 'DomainNet':
            self.save_folder_name = 'seen-' + args.seen_domain + '_unseen-' + args.holdout_domain + '_x_' + args.gallery_domain
            if not args.include_auxillary_domains:
                self.save_folder_name += '_noaux'
        elif args.dataset == 'Sketchy':
            if args.is_eccv_split:
                self.save_folder_name = 'eccv_split'
            else:
                self.save_folder_name = 'random_split'
        else:
            self.save_folder_name = ''

        if args.dataset == 'DomainNet' or (args.dataset == 'Sketchy' and args.is_eccv_split):
            self.map_metric = 'mAP@200'
            self.prec_metric = 'prec@200'
        else:
            self.map_metric = 'mAP@all'
            self.prec_metric = 'prec@100'

        self.suffix = '-e-' + str(args.epochs) + '_es-' + str(args.early_stop) + '_opt-' + args.optimizer + \
                      '_bs-' + str(args.batch_size) + '_lr-' + str(args.lr)

        # exit(0)
        self.path_cp = os.path.join(args.code_path, "src/algos/6_CoCoOp/log", args.dataset,
                                    self.save_folder_name)
        log_file = os.path.join(self.path_cp, args.log_name + ".txt")
        # Redirect print to both console and log file
        sys.stdout = Logger(log_file)

        self.start_epoch = 0
        self.best_map = 0
        self.early_stop_counter = 0
        self.last_chkpt_name = 'init'

        print("================Parameters Settings=================")
        print('Parameters:\t' + str(self.args))
        print("================Training Settings=================")
        print(f"lr = {self.args.lr}")
        print(f"batch_size = {self.args.batch_size}")
        print(f"visual prompt numbers = {self.args.vptNumTokens}")
        print(f"text prompt-setup = {self.args.text}")
        if self.args.text != 'None':
            print(f"text prompt numbers= {self.args.textNumTokens}")
        print(f"maple used？= {self.args.maple}")
        print(f"ivlp used？= {self.args.ivlp}")
        print("==================================================")

    def do_training(self):
        from thop import profile
        from torchvision.models import resnet50
        from torchstat import stat
        model = resnet50().to(device)

        input = torch.randn(2, 3, 224, 224).to(device)
        mac, params = profile(self.model, inputs=(input,))
        print(f"FLOPS: {mac  / 1e9} G")
        print(f"Params: {params / 1e6} M")