from mail import mail
from local_utils import *
from src.algos.basetrainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.set_model()
        self.show()

    def set_model(self):
        self.model = mail(self.args, self.dict_clss, self.dict_doms, device)
        self.model = self.model.to(device)

    def show(self):
        print("================Parameters Settings=================")
        print('Parameters:\t' + str(self.args))
        print("================Training Settings=================")
        print(f"lr = {self.args.lr}")
        print(f"batch_size = {self.args.batch_size}")
        print("==================================================")

    def set_trainable_parameters(self) -> list:
        train_parameters = ['mail_learner', 'last_adapter', 'last_ln']
        if self.args.proj:
            train_parameters.append('text_encoder.text_projection')
            train_parameters.append('visual_encoder.proj')
        return train_parameters

    def visual_forward(self, im):
        prompts, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp, att_proj_layers_att = self.model.mail_learner()
        out = self.model.visual_encoder(im, ln_proj_layers_mlp, ln_proj_layers_att, mlp_proj_layers_mlp,
                                        att_proj_layers_att, self.model.last_ln)
        if self.model.last_adapter is not None:
            out = self.model.last_adapter(out, is_text=False, i=12)
        return out
