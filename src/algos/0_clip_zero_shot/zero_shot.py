import argparse
import sys
import os
from tqdm import tqdm
import os

code_path = '/home/user/Code/MAIL'
data_path = '/data/UCDR/data'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))

import torch
import  clip
from torch.utils.data import DataLoader
from src.data.DomainNet import domainnet
from src.data.Sketchy import sketchy_extended
from src.data.TUBerlin import tuberlin_extended

import numpy as np
import torch.backends.cudnn as cudnn
from src.data.dataloaders import BaselineDataset
from src.utils import utils, GPUmanager
from src.utils.metrics import compute_retrieval_metrics
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')



class Tester:

    def __init__(self, args):
        self.args = args
        print('\nLoading data...')
        if args.dataset == 'DomainNet':
            data_input = domainnet.create_trvalte_splits(args)
        if args.dataset == 'Sketchy':
            data_input = sketchy_extended.create_trvalte_splits(args)
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

        self.model, self.preprocess = clip.load('ViT-B/32', device, jit=False)
        # Image transformations
        self.image_transforms = {
            'eval': self.preprocess
        }

        # class dictionary
        self.te_dict_class = utils.create_dict_texts(self.tr_classes+self.va_classes+self.te_classes)

        fls_tr = self.data_splits['tr'] # train image paths
        dom_tr = np.array([f.split('/')[-3] for f in fls_tr]) # train image dom
        tr_domains_unique = np.unique(dom_tr)

        # doamin dictionary
        self.dict_doms = utils.create_dict_texts(tr_domains_unique)
        print(self.dict_doms)

        self.model = self.model.to(device)

        if args.dataset=='DomainNet' or (args.dataset=='Sketchy' and args.is_eccv_split):
            self.map_metric = 'mAP@200'
            self.prec_metric = 'prec@200'
        else:
            self.map_metric = 'mAP@all'
            self.prec_metric = 'prec@100'

    def test(self):    
        te_data = []
        if self.args.dataset == 'DomainNet':
            if self.args.udcdr == 0:
                for domain in [self.args.holdout_domain]:
                    for includeSeenClassinTestGallery in [0,1]:
                        test_head_str = 'Query:' + domain + '; Gallery:' + self.args.gallery_domain + '; Generalized:' + str(includeSeenClassinTestGallery)
                        print(test_head_str)
                        
                        splits_query = domainnet.trvalte_per_domain(self.args, domain, 0, self.tr_classes,  self.te_classes)
                        splits_gallery = domainnet.trvalte_per_domain(self.args, self.args.gallery_domain, includeSeenClassinTestGallery, self.tr_classes, self.te_classes)

                        data_te_query = BaselineDataset(np.array(splits_query['te']), transforms=self.image_transforms['eval'])
                        data_te_gallery = BaselineDataset(np.array(splits_gallery['te']), transforms=self.image_transforms['eval'])

                        # PyTorch test loader for query
                        te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 10, shuffle=False,
                                                        num_workers=self.args.num_workers, pin_memory=True)
                        # PyTorch test loader for gallery
                        te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 10, shuffle=False,
                                                        num_workers=self.args.num_workers, pin_memory=True)

                        result = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4, self.args)
                        te_data.append(result)
                        
                        out = f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(result[self.map_metric], result[self.prec_metric])
                        print(out)
            else:
               
                if self.args.holdout_domain == 'quickdraw':
                    p = 0.1
                else :
                    p = 0.25
                splits_query = domainnet.seen_cls_te_samples(self.args, self.args.holdout_domain, self.tr_classes, p)
                splits_gallery = domainnet.seen_cls_te_samples(self.args, self.args.gallery_domain, self.tr_classes, p)

                data_te_query = BaselineDataset(np.array(splits_query), transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(np.array(splits_gallery), transforms=self.image_transforms['eval'])

                # PyTorch test loader for query
                te_loader_query = DataLoader(dataset=data_te_query, batch_size=800, shuffle=False,
                                                num_workers=self.args.num_workers, pin_memory=True)
                # PyTorch test loader for gallery
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=800, shuffle=False,
                                                num_workers=self.args.num_workers, pin_memory=True)
                result = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4, self.args)
                map_ = result[self.map_metric]
                prec = result[self.prec_metric]
                out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(map_, prec)
                print(out)
        else :
            data_te_query = BaselineDataset(self.data_splits['query_te'], transforms=self.image_transforms['eval'])
            data_te_gallery = BaselineDataset(self.data_splits['gallery_te'], transforms=self.image_transforms['eval'])

            te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
            te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size*5, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

            print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

            te_data = evaluate(te_loader_query, te_loader_gallery, self.model,self.te_dict_class, self.dict_doms, 4, self.args)
            out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(te_data[self.map_metric], te_data[self.prec_metric])
        
            map_ = te_data[self.map_metric]
            prec = te_data[self.prec_metric]
            out =f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n"%(map_, prec)
            print(out)


@torch.no_grad()
def evaluate(loader_sketch, loader_image, model, dict_clss, dict_doms, stage, args):

    # Switch to test mode
    model.eval()

    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in tqdm(enumerate(loader_sketch), desc='Extrac query feature', total=len(loader_sketch)):

        sk = sk.float().to(device)
        cls_id = utils.numeric_classes(cls_sk, dict_clss)
        sk_em = model.encode_image(sk)
        sketchEmbeddings.append(sk_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        sketchLabels.append(cls_numeric)
        if  args.debug_mode == 1 and i == 2:
            break
    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in tqdm(enumerate(loader_image), desc='Extrac gallery feature', total=len(loader_image)):

        im = im.float().to(device)
        cls_id = utils.numeric_classes(cls_im, dict_clss)
        dom_id = utils.numeric_classes(dom, dict_doms)
        # Clipart embedding into a semantic space
        im_em = model.encode_image(im)
        realEmbeddings.append(im_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        realLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

    return eval_data

class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='UCDR_MAPLE')
        parser.add_argument('-log_name', '--log_name', type=str, default='log',
                            help='log name :)')

        parser.add_argument('-resume', '--resume_dict', type=str, help='checkpoint file to resume training from')

        # data_root

        parser.add_argument('-code_path', '--code_path', default=code_path, type=str, help='code path of UCDR')
        parser.add_argument('-dataset_path', '--dataset_path', default=data_path, type=str,
                            help='Path of three datasets')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet', 'TUBerlin'])
        parser.add_argument('-eccv', '--is_eccv_split', choices=[0, 1], default=1, type=int, help='whether or not to use eccv18 split\
                            if dataset="Sketchy"')
        # CLIP
        parser.add_argument('-clip_bb', '--clip_backbone', type=str,
                            choices=['RN50x4', 'RN50x16', 'ViT-B/16', 'ViT-B/32'], default='ViT-B/32',
                            help='choose clip backbone')

        parser.add_argument('-debug_mode', '--debug_mode', default=0, type=int, help='use debug model')

        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='painting',
                            choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-hd', '--holdout_domain', default='infograph',
                            choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-gd', '--gallery_domain', default='real',
                            choices=['clipart', 'infograph', 'photo', 'painting', 'real'])
        parser.add_argument('-aux', '--include_auxillary_domains', choices=[0, 1], default=1, type=int,
                            help='whether(1) or not(0) to include\domains other than seen domain and gallery')
        parser.add_argument('-udcdr', '--udcdr', choices=[0, 1], default=0, type=int,
                            help='whether or not to evaluate under udcdr protocol')

        # Size parameters
        parser.add_argument('-imsz', '--image_size', default=224, type=int,
                            help='Input size for query/gallery domain sample')

        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=50, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of workers in data loader')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=2, help='Early stopping epochs.')

        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()

def main(args):
    trainer = Tester(args)
    trainer.test()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)