import sys
import time
from tqdm import tqdm
import os

from src.losses.sup_con_loss import soft_sup_con_loss, triplet_loss
from src.utils import utils
from src.utils.logger import AverageMeter

code_path = '/home/user/Code/DePro' # e.g. '/home/username/ProS'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))
sys.path.append(os.path.join(code_path, "clip"))
sys.path.append(os.path.join(code_path, "loralib"))

from loralib import apply_lora, mark_only_lora_as_trainable
from clip import clip

from torch import optim



from local_utils import *
from src.algos.basetrainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.set_model()
        self.show()

    def set_model(self):
        self.model, self.preprocess = clip.load(self.args.clip_backbone)
        self.model.eval()

    def training_set(self):
        tot = 0
        for name, param in self.model.named_parameters():
            tot += param.numel()
        lr = self.args.lr

        list_lora_layers = apply_lora(self.args, self.model)
        self.model = self.model.to(device)
        self.model.float()
        mark_only_lora_as_trainable(self.model)

        print("======== Context-aware Simulator Learning Setup========")
        train_part = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
                train_part += param.numel()
            if name in ['text_projection', 'visual.proj']:
                param.requires_grad_(True)  # no if ucdr

        print(f"tot={tot}, train = {train_part} (with no proj)")
        # NOTE: only give prompt_learner to the optimizer
        optimizer = None
        if self.args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                  weight_decay=self.args.l2_reg, momentum=self.args.momentum, nesterov=False, lr=lr)
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                   betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.l2_reg)
        print("===============================================")
        return optimizer

    def visual_forward(self, im):
        return self.model.visual(im)

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
        classnames = self.dict_clss.keys()
        prompt_prefix = self.args.ctx_init

        for i, (im, cls, dom) in enumerate(train_loader):

            im = im.float().to(device, non_blocking=True)
            cls_numeric = torch.from_numpy(utils.numeric_classes(cls, self.dict_clss)).long().to(device)
            optimizer.zero_grad()


            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
            with torch.no_grad():
                class_embeddings = self.model.encode_text(tokenized_prompts).to(device)

            soft_label = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            feature = self.model.visual(im)

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
        return {'net': losss.avg, 'acc': correct / tot}

