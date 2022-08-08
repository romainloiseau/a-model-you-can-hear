import numpy as np
import time
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from tqdm.auto import tqdm
import copy
from .logging import LoggingModel


class BaseModel(pl.LightningModule, LoggingModel):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.__initmodel__(*args, **kwargs)
        self.__initmetrics__()
        self.__compute_nparams__()

        if self.hparams.log_graph:
            self.example_input_array = torch.randn((self.hparams.data.batch_size, 1, self.hparams.data.n_mels, self.hparams.data.train_spect_size))

    def __compute_nparams__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def __initmetrics__(self):
        self.lambda_reconstruction = copy.deepcopy(self.hparams.lambda_reconstruction)
        self.lambda_crossentropy = copy.deepcopy(self.hparams.lambda_crossentropy)
        return
        
    def __initmodel__(self, *args, **kwargs):
        self.noise = nn.Parameter(torch.ones((1, )))

    def __initmask__(self, *args, **kwargs):
        return

    def greedy_model(self):
        raise NotImplementedError

    def forward(self, batch, *args, **kwargs):
        with torch.profiler.record_function(f"FORWARD"):
            return batch + self.noise, None, None

    def on_train_start(self):
        results = {
            "Loss/val": float("nan"),
            "Loss/train": float("nan"),
        }

        self.logger.log_hyperparams(self.hparams, results)

        self.logger.experiment.add_text(
            "model", self.__repr__().replace("\n", "  \n"), global_step=0)

    def global_step(self, batch, batch_idx, tag):
        out = self.forward(batch, batch_idx, tag)

        with torch.profiler.record_function(f"LOSS"):
            loss = self.lambda_reconstruction * out.recerror
            if self.lambda_crossentropy != 0:
                loss = loss + self.lambda_crossentropy * out.crossentropy

        with torch.no_grad():
            with torch.profiler.record_function(f"LOGGERS"):

                self.log(f'Loss/{tag}',
                            loss, on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))

                self.log(f'Losses/rec/{tag}',
                            out.recerror, on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))

                self.log(f'Losses/ce/{tag}',
                            out.crossentropy, on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))

                self.greedy_step(batch["spectrogram"], out.rec, batch_idx, tag)

                if tag == "train":
                    self.update_count(out.choice)                
                         
        return {"loss": loss, "choice": out.choice, "layer": out.layer}

    def update_count(self, choice):
        return

    def training_step(self, batch, batch_idx):
        return self.global_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.global_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.global_step(batch, batch_idx, 'test')

    def log_step(self):
        self.log(f'step', 1. + self.current_epoch, on_step=False, on_epoch=True)

    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)

        self.log_step()

    def on_validation_epoch_end(self, *args, **kwargs):
        super().on_validation_epoch_end(*args, **kwargs)

        self.log_step()

    def on_test_epoch_end(self, *args, **kwargs):
        super().on_test_epoch_end(*args, **kwargs)

        self.log_step()

    def configure_optimizers(self):
        parameters = [
                {"params": [], "name": "base"},
                {"params": self._protos, "name": f"protos", 'weight_decay': 0},
                {"params": self.ce_temperature, "name": f"ce_temperature", 'weight_decay': 0, "lr": self.hparams.lr_scale_ce_temperature*self.hparams.optim.lr},
                {"params": self.encoder.parameters(), "name": "encoder"}
        ]

        for decoder in self.decoders:
            parameters.append({
                "params": self.decoders[decoder].parameters(),
                "name": decoder
            })

        optimizer = getattr(torch.optim, self.hparams.optim._target_)(
            parameters,
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay
        )

        return {
            "optimizer": optimizer,
        }

