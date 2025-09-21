from mmadapter import mma
from local_utils import *
from src.algos.basetrainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.set_model()
        self.show()

    def set_model(self):
        self.model = mma(self.args, self.dict_clss, self.dict_doms, device)
        self.model = self.model.to(device)

    def show(self):
        print("================Parameters Settings=================")
        print('Parameters:\t' + str(self.args))


    def set_trainable_parameters(self) -> list:
        train_parameters = ['text_prompt_learner']  # 'feature_template'
        if self.args.proj:
            train_parameters.append('text_encoder.text_projection')
            train_parameters.append('visual_encoder.proj')
        return train_parameters

    def visual_forward(self, im):
        prompts, text_adapter_func, visual_adapter_func = self.model.text_prompt_learner()
        out = self.model.visual_encoder([im, visual_adapter_func])
        return out



