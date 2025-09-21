import os
import sys
import math
import time
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from src.data.dataloaders import CuMixloader, BaselineDataset
from src.data.sampler import FewShotSampler
from src.utils.logger import AverageMeter
from src.losses.sup_con_loss import soft_sup_con_loss, triplet_loss
from src.utils.logging import Logger

from mail import mail
from local_utils import *


class Trainer:
    def __init__(self, args):
        self.args = args
        print('\nLoading data...')
        self.tr_classes, self.te_classes, self.data_splits = get_data_info(args, args.dataset)
        # the above are all texts :)
        self.dict_clss = utils.create_dict_texts(self.tr_classes)  # 生成 类:index 的一个字典 word to digit
        self.te_dict_class = utils.create_dict_texts(self.tr_classes + self.te_classes)
        fls_tr = self.data_splits['tr']
        cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
        dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
        tr_domains_unique = np.unique(dom_tr)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            cudnn.benchmark = True  # cant' guarantee reproduce for faster training
            torch.cuda.manual_seed_all(args.seed)

        self.image_transforms = get_transform(self.args.image_size)

        # doamin dictionary
        self.dict_doms = utils.create_dict_texts(tr_domains_unique)

        domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)

        data_train = CuMixloader(fls_tr, cls_tr, dom_tr, self.dict_doms, transforms=self.image_transforms['train'])

        cls_dis = utils.numeric_classes(cls_tr, self.dict_clss)

        train_sampler = FewShotSampler(domain_ids, cls_dis,
                                       domains_per_batch=len(tr_domains_unique),
                                       num_shots=self.args.num_shots)  # 每个batch的采样都来自同一个domain

        self.train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)

        print('Loading Done\n')

        self.model = mail(self.args, self.dict_clss, self.dict_doms, device)
        self.model = self.model.to(device)

        self.save_folder_name = get_folder_name(args)

        self.map_metric = 'mAP@all'
        self.prec_metric = 'prec@100'
        if args.dataset == 'DomainNet' or (args.dataset == 'Sketchy' and args.is_eccv_split):
            self.map_metric = 'mAP@200'
            self.prec_metric = 'prec@200'


        self.suffix = '-e-' + str(args.epochs) + '_es-' + str(args.early_stop) + '_opt-' + args.optimizer + \
                      '_bs-' + str(args.batch_size) + '_lr-' + str(args.lr)

        # exit(0)
        self.path_cp = os.path.join(os.getcwd(), "log", args.dataset,
                                    self.save_folder_name)

        log_file = os.path.join(self.path_cp, args.log_name + ".txt")

        # Redirect print to both console and log file
        sys.stdout = Logger(log_file)
        self.start_epoch = 0
        self.best_map = 0
        self.early_stop_counter = 0
        self.last_ckpt_name = 'init'

        print("================Parameters Settings=================")
        print('Parameters:\t' + str(self.args))
        print("================Training Settings=================")
        print(f"lr = {self.args.lr}")
        print(f"batch_size = {self.args.batch_size}")
        print("==================================================")

    def training_set(self):
        tot = 0
        for name, param in self.model.named_parameters():
            tot += param.numel()
        lr = self.args.lr
        print("======== The list of trainable parameters of MAIL========")
        train_parameters = ['mail_learner', 'last_adapter', 'last_ln']
        if self.args.proj:
            train_parameters.append('text_encoder.text_projection')
            train_parameters.append('visual_encoder.proj')

        train_part = 0
        for name, param in self.model.named_parameters():
            for str in train_parameters:
                flag = 0
                if name.startswith(str) == True:
                    param.requires_grad_(True)
                    if str not in ['text_encoder.text_projection', 'visual_encoder.proj']:
                        train_part += param.numel()
                    flag = 1
                    break
            if flag == 0:
                param.requires_grad_(False)
            else:
                print(name)
        print(f"tot={tot}, train = {train_part} (with no proj)")

        optimizer = None
        if self.args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                  weight_decay=self.args.l2_reg,
                                  momentum=self.args.momentum, nesterov=False, lr=lr)
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-8, weight_decay=self.args.l2_reg)
        print("===============================================")
        return optimizer

    def post_precess(self, result_unnorm, result_norm):
        map_unnorm = result_unnorm[self.map_metric]
        map_norm = result_norm[self.map_metric]

        prec_unnorm = result_unnorm[self.prec_metric]
        prec_norm = result_norm[self.prec_metric]

        print("==========The results of un-normed situation: ==========")
        print(f" {self.map_metric}: {map_unnorm:.2%}, {self.prec_metric}: {prec_unnorm:.2%}")

        print("==========The results of normed situation: ==========")
        print(f" {self.map_metric}: {map_norm:.2%}, {self.prec_metric}: {prec_norm:.2%}")

        return max(map_norm, map_unnorm), max(prec_norm, prec_unnorm)

    def do_training(self):
        optimizer = self.training_set()
        for current_epoch in range(self.start_epoch, self.args.epochs):
            start = time.time()
            self.adjust_learning_rate(optimizer, current_epoch)
            loss = self.do_epoch(optimizer, current_epoch)
            print(f"Epoch = [{current_epoch + 1}/{self.args.epochs}] Loss = {loss}")

            if self.args.dataset == 'DomainNet':
                domain = self.args.holdout_domain
                print(
                    f"\n==================================1. Evaluating {self.args.num_shots}-shot UCDR: "
                    f"==================================")

                for includeSeenClassinTestGallery in [0, 1]:
                    test_head_str = 'Query:' + domain + '; Gallery:' + self.args.gallery_domain + '; Mixed Gallery:' + \
                                    str(bool(includeSeenClassinTestGallery))
                    test_head_str = test_head_str if not includeSeenClassinTestGallery else "\n" + test_head_str
                    print(test_head_str)

                    splits_query = domainnet.trvalte_per_domain(self.args, domain, 0, self.tr_classes,
                                                                self.te_classes)
                    splits_gallery = domainnet.trvalte_per_domain(self.args, self.args.gallery_domain,
                                                                  includeSeenClassinTestGallery, self.tr_classes,
                                                                  self.te_classes)

                    data_te_query = BaselineDataset(np.array(splits_query['te']), transforms=self.image_transforms['eval'])
                    data_te_gallery = BaselineDataset(np.array(splits_gallery['te']), transforms=self.image_transforms['eval'])

                    # PyTorch test loader for query
                    te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 10,
                                                 shuffle=False,
                                                 num_workers=self.args.num_workers, pin_memory=True)
                    # PyTorch test loader for gallery
                    te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 10,
                                                   shuffle=False,
                                                   num_workers=self.args.num_workers, pin_memory=True)

                    result_unnorm, result_norm \
                        = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.args)
                    map_, prec_ = self.post_precess(result_unnorm, result_norm)

                print(
                    f"\n==================================2. Evaluating {self.args.num_shots}-shot U^dCDR: ====="
                    f"=============================")

                p = 0.1 if self.args.holdout_domain == 'quickdraw' else 0.25
                splits_query = domainnet.seen_cls_te_samples(self.args, self.args.holdout_domain, self.tr_classes, p)
                splits_gallery = domainnet.seen_cls_te_samples(self.args, self.args.gallery_domain, self.tr_classes, p)

                data_te_query = BaselineDataset(np.array(splits_query), transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(np.array(splits_gallery), transforms=self.image_transforms['eval'])

                # PyTorch test loader for query
                te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 5, shuffle=False,
                                             num_workers=self.args.num_workers, pin_memory=True)
                # PyTorch test loader for gallery
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 5,
                                               shuffle=False,
                                               num_workers=self.args.num_workers, pin_memory=True)
                result_unnorm, result_norm \
                    = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.args)
                _, _ = self.post_precess(result_unnorm, result_norm)

            else:
                print(
                    f"\n==================================1. Evaluating {self.args.num_shots}-shot U^cCDR: ====="
                    f"=============================")
                data_te_query = BaselineDataset(self.data_splits['query_te'], transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(self.data_splits['gallery_te'],
                                                  transforms=self.image_transforms['eval'])

                te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 5, shuffle=False,
                                             num_workers=self.args.num_workers, pin_memory=True)
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 5,
                                               shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

                print(
                    f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

                result_unnorm, result_norm \
                    = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.args)
                map_, prec_ = self.post_precess(result_unnorm, result_norm)

            end = time.time()
            elapsed = end - start

            print(
                f"Epoch Time:{elapsed // 60:.0f}m{elapsed % 60:.0f}s lr:{utils.get_lr(optimizer):.7f} mAP:{map_:.4f} prec:{prec_:.4f}\n")

            if map_ > self.best_map:
                self.best_map = map_
                self.early_stop_counter = 0
                model_save_name = 'val_map-' + '{0:.4f}'.format(map_)
                utils.save_checkpoint({
                    'epoch': current_epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_map': self.best_map,
                }, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_ckpt_name)
                self.last_ckpt_name = model_save_name
            else:
                self.early_stop_counter += 1
                if self.args.early_stop == self.early_stop_counter:
                    print(f"Validation Performance did not improve for {self.args.early_stop} epochs."
                          f"Early stopping by {self.args.epochs - current_epoch - 1} epochs.")
                    break
                print(f"Val mAP hasn't improved from {self.best_map:.4f} for {self.early_stop_counter} epoch(s)!\n")
        print('\n***Training and Validation complete***')

    def adjust_learning_rate(self, optimizer, current_epoch, min_lr=1e-6, ):
        lr = self.args.lr * math.pow(1e-3, float(current_epoch) / 20)
        lr = max(lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def resume_from_checkpoint(self, resume_dict):
        if resume_dict is not None:
            print('==> Resuming from checkpoint: ', resume_dict)
            model_path = os.path.join(self.path_cp, resume_dict + '.pth')
            checkpoint = torch.load(model_path, map_location=device)
            self.start_epoch = checkpoint['epoch'] + 1
            # self.last_chkpt_name = resume_dict
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # self.best_map = checkpoint['best_map']

    def do_epoch(self, optimizer, current_epoch):
        self.model.train()

        batch_time = AverageMeter()

        loss_clss = AverageMeter()
        loss_triplets = AverageMeter()
        losss = AverageMeter()
        # Start counting time
        time_start = time.time()

        train_loader = self.train_loader
        correct = 0
        tot = 0
        for i, (im, cls, dom) in enumerate(train_loader):
            im = im.float().to(device, non_blocking=True)
            cls_numeric = torch.from_numpy(utils.numeric_classes(cls, self.dict_clss)).long().to(device)
            optimizer.zero_grad()

            feature, soft_label = self.model(im, dom, cls)

            hard_labels = cls_numeric
            loss_cls, co = soft_sup_con_loss(feature, soft_label, hard_labels, device=device)
            loss_triplet, co2 = triplet_loss(feature, hard_labels)
            correct += co
            tot += im.size(0)
            loss = loss_cls + loss_triplet
            loss.backward()
            optimizer.step()

            losss.update(loss.item(), im.size(0))
            loss_clss.update(loss_cls.item(), im.size(0))
            loss_triplets.update(loss_triplet.item(), im.size(0))

            # time
            time_end = time.time()
            batch_time.update(time_end - time_start)
            time_start = time_end

            if (i + 1) % self.args.log_interval == 0:
                print('[Train] Epoch: [{0}/{1}][{2}/{3}]  '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'cls {net1.val:.4f} ({net1.avg:.4f})  '
                      'contrastive {net2.val:.4f} ({net2.avg:.4f})  '
                      'loss {net3.val:.4f} ({net3.avg:.4f})  '
                      .format(current_epoch + 1, self.args.epochs, i + 1, len(train_loader), batch_time=batch_time,
                              net1=loss_clss, net2=loss_triplets, net3=losss))
                if self.args.debug_mode == 1:
                    break
        return {'net': losss.avg, 'acc': correct / (tot + 1)}
