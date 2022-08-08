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


class Encoder(nn.Module):
    def __init__(
            self,
            start_dim,
            n_filters=(32, 32, 32),
            kernel_size=(5, 5),
            padding_mode='zeros',
            dilation=(1, 1),
            pool_size=(2, 2)):

        super(Encoder, self).__init__()

        conv1 = nn.Conv2d(start_dim, n_filters[0], kernel_size=kernel_size, padding=(2, 2), padding_mode=padding_mode, dilation=dilation)
        act1 = nn.LeakyReLU()
        self.conv1 = nn.Sequential(*[conv1, act1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=tuple(pool_size))
        

        conv2 = nn.Conv2d(n_filters[0], n_filters[1], kernel_size=kernel_size, padding=(2, 2), padding_mode=padding_mode, dilation=dilation)
        act2 = nn.LeakyReLU()
        self.conv2 = nn.Sequential(*[conv2, act2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=tuple(pool_size))

        conv3 = nn.Conv2d(n_filters[1], n_filters[2], kernel_size=kernel_size, padding=(2, 2), padding_mode=padding_mode, dilation=dilation)
        act3 = nn.LeakyReLU()
        self.conv3 = nn.Sequential(*[conv3, act3])
        self.maxpool3 = nn.MaxPool2d(kernel_size=tuple(pool_size))

        conv4 = nn.Conv2d(n_filters[2], n_filters[3], kernel_size=kernel_size, padding=(2, 2), padding_mode=padding_mode, dilation=dilation)
        act4 = nn.LeakyReLU()
        self.conv4 = nn.Sequential(*[conv4, act4])

    def forward(self, x):
        y = self.conv1(x)
        y_pool = self.maxpool1(y)

        y = self.conv2(y_pool)
        y_pool = self.maxpool2(y)

        y = self.conv3(y_pool)
        y_pool = self.maxpool3(y)

        y = self.conv4(y_pool)
        return y

class Decoder(nn.Module):
    def __init__(
            self,
            start_dim,
            n_filters=(32, 32, 32),
            kernel_size=(5, 5),
            padding_mode='zeros',
            dilation=(1, 1),
            pool_size=(2, 2)):

        super(Decoder, self).__init__()

        self.act = nn.LeakyReLU()

        conv1 = nn.ConvTranspose2d(n_filters[3], n_filters[2], kernel_size=kernel_size, padding=(2, 2), padding_mode="zeros", dilation=dilation)
        upsample1 = nn.Upsample(scale_factor=tuple(pool_size))
        self.deconv1 = nn.Sequential(*[conv1, upsample1])

        conv2 = nn.ConvTranspose2d( n_filters[2], n_filters[1], kernel_size=kernel_size, padding=(2, 2), padding_mode="zeros", dilation=dilation)
        upsample2 = nn.Upsample(scale_factor=tuple(pool_size))
        self.deconv2 = nn.Sequential(*[conv2, upsample2])

        conv3 = nn.ConvTranspose2d( n_filters[1], n_filters[0], kernel_size=kernel_size, padding=(2, 2), padding_mode="zeros", dilation=dilation)
        upsample3 = nn.Upsample(scale_factor=tuple(pool_size))
        self.deconv3 = nn.Sequential(*[conv3, upsample3])

        conv4 = nn.ConvTranspose2d(n_filters[0], start_dim, kernel_size=kernel_size, padding=(2, 2), padding_mode="zeros", dilation=dilation)
        act4 = nn.Tanh()
        self.deconv4 = nn.Sequential(*[conv4, act4])

    def forward(self, x):
        deconv = self.act(self.deconv1(x))
        deconv = self.act(self.deconv2(deconv))
        deconv = self.act(self.deconv3(deconv))
        deconv = self.deconv4(deconv)
        return deconv


class AutoEncoder(BaseModel):
    def __initmetrics__(self):
        self.loss = lambda inp, out: torch.mean(torch.mean(torch.pow(inp - out, 2), dim=(1, 2, 3)))

    def __initmodel__(self, *args, **kwargs):

        self.encoder = Encoder(
            self.hparams.start_dim,
            self.hparams.n_filters,
            self.hparams.kernel_size,
            self.hparams.padding_mode,
            self.hparams.dilation,
            self.hparams.pool_size
        )

        self.decoder = Decoder(
            self.hparams.start_dim,
            self.hparams.n_filters,
            self.hparams.kernel_size,
            self.hparams.padding_mode,
            self.hparams.dilation,
            self.hparams.pool_size
        )

    def greedy_model(self):
        return

    def get_frequency_weights(self):
        return torch.softmax(self.frequency_weights, dim=1)

    @torch.profiler.record_function(f"FORWARD")
    def forward(self, batch, *args, **kwargs):
        # batch["spectrogram"] is B*C*F*T
        latent_features = self.encoder(2*batch["spectrogram"]-1)
        # latent_features is B*Cfeat*F*T
        reconstructed_input = .5* (1+self.decoder(latent_features))
        return reconstructed_input, latent_features

    def global_step(self, batch, batch_idx, tag):
        reconstruction, latent_features = self.forward(batch, batch_idx, tag)

        with torch.profiler.record_function(f"LOSS"):
            loss = self.loss(batch["spectrogram"], reconstruction)

        with torch.no_grad():
            with torch.profiler.record_function(f"LOGGERS"):
                self.log(f'Loss/{tag}',
                         loss, on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))
                self.log(f'Losses/rec/{tag}', loss, 
                    on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))


                self.greedy_step(batch["spectrogram"], reconstruction, batch_idx, tag)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optim._target_)(
            self.parameters(),
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay
        )

        return {
            "optimizer": optimizer,
        }