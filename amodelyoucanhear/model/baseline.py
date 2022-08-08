from matplotlib.pyplot import xlim
import numpy as np
import time
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from tqdm.auto import tqdm

from .logging import LoggingModel

from .base import BaseModel

def getSpectEncoder2D(start_dim, n_pools, padding_mode):
    layers = []
        
    for i in range(n_pools):
        dim_in, dim_out = 1 if i == 0 else start_dim*2**(i-1), start_dim*2**i
            
        layers += [
            nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), padding=(1, 1), padding_mode=padding_mode),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out),
            nn.Conv2d(dim_out, dim_out, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), padding_mode=padding_mode),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)
        ]

    return nn.Sequential(*layers) 

def getLinearDecoder(start_dim, decoder, out_dim):
    layers = []
        
    for i in range(len(decoder)):
        dim_in, dim_out = start_dim if i == 0 else decoder[i-1], decoder[i]
            
        layers += [
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.BatchNorm1d(dim_out)
        ]

    layers += [nn.Linear(decoder[-1], out_dim)]

    return nn.Sequential(*layers) 


class BaselineModel(BaseModel):
    def __initmetrics__(self):
        self.loss = nn.CrossEntropyLoss()

    def __initmodel__(self, *args, **kwargs):
        self.encoder_conv = getSpectEncoder2D(
            self.hparams.start_dim,
            self.hparams.n_pools,
            self.hparams.padding_mode
        )
        dim_linear = int(self.hparams.start_dim * self.hparams.data.n_mels / 2)

        self.decoder = getLinearDecoder(dim_linear, self.hparams.decoder, self.hparams.K)

    def greedy_model(self):
        return

    @torch.profiler.record_function(f"FORWARD")
    def forward(self, batch, *args, **kwargs):
        # batch["spectrogram"] is B*C*F*T
        encoded = self.encoder_conv(batch["spectrogram"])
        encoded = encoded.max(-1)[0].flatten(-2, -1)
        prediction = self.decoder(encoded)
        return prediction # is B*K

    def global_step(self, batch, batch_idx, tag):
        out = self.forward(batch, batch_idx, tag)

        with torch.profiler.record_function(f"LOSS"):
            loss = self.loss(out, batch['label'])

        with torch.no_grad():
            with torch.profiler.record_function(f"LOGGERS"):
                
                self.log(f'Loss/{tag}',
                            loss, on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))

                self.greedy_step(batch["spectrogram"], None, batch_idx, tag)     
                         
        return {"loss": loss, "choice": out.argmax(-1).unsqueeze(-1)}

    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.hparams.optim._target_)(
            self.parameters(),
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay
        )

        return {
            "optimizer": optimizer,
        }