from maple import maple
from local_utils import *
from src.algos.basetrainer import BaseTrainer

# testing packages
import torch
from tqdm import tqdm
from src.utils import utils
from src.utils.metrics import compute_retrieval_metrics


class Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.set_model()
        self.show()

    def set_model(self):
        self.model = maple(self.args, self.dict_clss, self.dict_doms, device)
        self.model = self.model.to(device)

    def show(self):
        print("================Parameters Settings=================")
        print('Parameters:\t' + str(self.args))
        print("================Training Settings=================")
        print(f"lr = {self.args.lr}")
        print(f"batch_size = {self.args.batch_size}")
        if self.args.maple:
            print("MaPLe used.")
            print(f"prompts depth = {self.args.maple_depth}")
            print(f"prompts length = {self.args.maple_length}")
        else:
            print("IVLP used.")
        print("==================================================")

    def set_trainable_parameters(self) -> list:
        train_parameters = ['text_prompt_learner']  # 'feature_template'
        if self.args.proj:
            train_parameters.append('text_encoder.text_projection')
            train_parameters.append('visual_encoder.proj')
        return train_parameters

    def visual_forward(self, im):
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.model.text_prompt_learner()
        out = self.model.visual_encoder(im, shared_ctx, deep_compound_prompts_vision)
        return out



