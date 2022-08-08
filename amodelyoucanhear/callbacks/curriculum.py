import torch
import copy
from tqdm.auto import tqdm

from .base import DTICallback

class Curriculum(DTICallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.monitor = self.myhparams.monitor
        self.patience = self.myhparams.patience
        self.last_value = None
        self.count = 0

        self.order = self.myhparams.order

        self.warmup = 0

        self.to_activate = None

        if self.myhparams.mode == "min":
            self.mode = -1
        elif self.myhparams.mode == "max":
            self.mode = 1
        else:
            raise ValueError(f"Argument 'mode' should be in ['min', 'max']\t'{self.myhparams.mode}' is invalid.")
    
    def on_train_start(self, trainer, pl_module):
        for transformation in self.order:
            if transformation == "CE":
                pl_module.lambda_crossentropy = 0
            elif "decay" not in transformation and transformation in pl_module.activated_transformations.keys():
                pl_module.activated_transformations[transformation] = False

            if transformation in pl_module.decoders.keys():
                torch.nn.init.normal_(
                    getattr(pl_module, "decoders")[transformation].regressor[-1].weight,
                    mean=self.myhparams.decoder_init_mean,
                    std=self.myhparams.decoder_init_std
                )
                torch.nn.init.normal_(
                    getattr(pl_module, "decoders")[transformation].regressor[-1].bias,
                    mean=self.myhparams.decoder_init_mean,
                    std=self.myhparams.decoder_init_std
                )

        for param_group in pl_module.optimizers().param_groups:
            if param_group["name"] in self.order:
                param_group['lr'] *= 0        

    def on_train_epoch_end(self, trainer, pl_module):
        if (len(self.order) != 0) and (self.to_activate is None):
            if self.last_value is None:
                self.last_value = trainer.callback_metrics[self.monitor].item()
            elif self.mode*trainer.callback_metrics[self.monitor].item() > self.mode*self.last_value:
                self.last_value = trainer.callback_metrics[self.monitor].item()
                self.count = 0
            else:
                self.count += 1

                if self.count >= (self.patience * (1 + int("decay" in self.order[0]))):
                    self.activate_transformation(trainer, pl_module)
                    self.last_value = None
                    self.count = 0

        for param_group in pl_module.optimizers().param_groups:
            pl_module.log(f'Learning_rate/{param_group["name"]}', param_group["lr"], on_step=False, on_epoch=True)
        pl_module.log(f'Lambda/crossentropy', pl_module.lambda_crossentropy, on_step=False, on_epoch=True)
        pl_module.log(f'Lambda/reconstruction', pl_module.lambda_reconstruction, on_step=False, on_epoch=True)
        
    def activate_transformation(self, trainer, pl_module):
        self.to_activate, self.order = self.order[0], self.order[1:]

        self.send_message(f"Activating {self.to_activate}", trainer.current_epoch)
        if self.myhparams.save_ckpt_at_activation:
            trainer.save_checkpoint(f"epoch={trainer.current_epoch}_{self.to_activate}.ckpt")

        if "decay" not in self.to_activate:
            if self.to_activate in pl_module.activated_transformations.keys():
                pl_module.activated_transformations[self.to_activate] = True
            self.warmup = copy.copy(self.myhparams.warmup_batch)

            if self.to_activate == "ce_temperature":
                to_augment = ["ce_temperature"]
            else:
                to_augment = [self.to_activate, "encoder"]

            for param_group in pl_module.optimizers().param_groups:
                
                if param_group["name"] in to_augment:
                    if param_group["name"] == "ce_temperature":
                        param_group['lr'] = pl_module.optimizers().param_groups[0]["lr"]*self.myhparams.warmup_intensity*pl_module.hparams.lr_scale_ce_temperature
                    else:
                        param_group['lr'] = pl_module.optimizers().param_groups[0]["lr"]*self.myhparams.warmup_intensity

        else:
            decay = float(self.to_activate.split("_")[1])
            for param_group in pl_module.optimizers().param_groups:
                param_group['lr'] = param_group['lr'] / decay
        
    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        if self.warmup > 0:
            self.warmup -= 1
            if self.to_activate == "CE":
                pl_module.lambda_crossentropy += (1. - self.myhparams.warmup_intensity) * pl_module.hparams.lambda_crossentropy / self.myhparams.warmup_batch
            

            if self.to_activate == "ce_temperature":
                for param_group in pl_module.optimizers().param_groups:
                    if param_group["name"] == "ce_temperature":
                        param_group['lr'] += (1. - self.myhparams.warmup_intensity)*pl_module.hparams.lr_scale_ce_temperature * pl_module.optimizers().param_groups[0]["lr"] / self.myhparams.warmup_batch
            else:
                for param_group in pl_module.optimizers().param_groups:
                    if param_group["name"] in [self.to_activate, "encoder"]:
                        param_group['lr'] += (1. - self.myhparams.warmup_intensity) * pl_module.optimizers().param_groups[0]["lr"] / self.myhparams.warmup_batch
                
        if self.warmup == 0:
            self.to_activate = None