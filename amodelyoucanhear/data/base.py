from types import SimpleNamespace
from typing import List, Tuple
from matplotlib.ft2font import KERNING_DEFAULT

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from torch import Tensor
from torch.utils.data import DataLoader
from torchaudio import transforms
from tqdm.auto import tqdm
import torch.nn.functional as F
from ..utils import play_audio
import copy
import math

from tqdm.auto import tqdm

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, myhparams, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.myhparams = copy.deepcopy(myhparams)
        self.mode = mode

        self.to_spect = transforms.Spectrogram(
            n_fft=self.myhparams.n_fft,
            win_length=self.myhparams.win_length,
            hop_length=self.myhparams.hop_length,
        )
        self.to_mel = transforms.MelScale(
            n_mels=self.myhparams.n_mels,
            sample_rate=self.myhparams.sample_rate,
            n_stft=1 + int(self.myhparams.n_fft/2.)
        )
        self.inv_mel = transforms.InverseMelScale(
            n_mels=self.myhparams.n_mels,
            sample_rate=self.myhparams.sample_rate,
            n_stft=1 + int(self.myhparams.n_fft/2.)
        )        
        self.inv_spect = transforms.GriffinLim(
            n_fft=self.myhparams.n_fft,
            win_length=self.myhparams.win_length,
            hop_length=self.myhparams.hop_length,
        )

    def get_to_mel(self, sample_rate=0):
        return self.to_mel

    def get_inv_mel(self, sample_rate=0):
        return self.inv_mel

    def __len__(self):
        return len(self.data)

    def wave2spect(self, wave, sample_rate=0):
        spect = self.to_spect(wave)
        spect = self.get_to_mel(sample_rate=sample_rate)(spect)
        logspect = (torch.log10(spect + 10**self.myhparams.log10_epsilon) - self.myhparams.log10_epsilon) / self.myhparams.log10_scale
        return logspect
        
    def spect2audio(self, logspect, sample_rate=0):
        spect = 10**(self.myhparams.log10_scale * logspect + self.myhparams.log10_epsilon)
        if spect.size(-1) == 1:
            spect = spect.repeat(1, 1, 1, 5)
        with torch.enable_grad():
            spect = self.get_inv_mel(sample_rate=sample_rate).to(spect.device)(spect.detach()).cpu()
        return self.inv_spect(spect)

    def restrict_spect(self, spect):
        if spect.size(-1) > self.myhparams.train_spect_size:
            start = np.random.randint(spect.size(-1) - self.myhparams.train_spect_size)
            return spect[..., start:start+self.myhparams.train_spect_size]
        elif spect.size(-1) == self.myhparams.train_spect_size:
            return spect
        else:
            return torch.cat([spect for _ in range(1+int(self.myhparams.train_spect_size / spect.size(-1)))], -1)[..., :self.myhparams.train_spect_size]

    def augment_spect(self, spect):
        if self.myhparams.augmentation_pitch != 0:
            C, H, T = spect.shape

            theta = torch.tensor([[1., 0, 0], [0, 1, 0]]).unsqueeze(0)
            scale = torch.exp(self.myhparams.augmentation_pitch * torch.randn((1, )))
            theta[:, 1, 1] = scale
            theta[:, 1, 2] = scale - 1.
            grid = torch.nn.functional.affine_grid(theta, (1, C, H, T), align_corners=False)
            spect = torch.nn.functional.grid_sample(spect.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False)[0]

        if self.myhparams.augmentation_bias != 0:
            spect = spect + self.myhparams.augmentation_bias * torch.randn((1, ))

        if self.myhparams.augmentation_noise != 0:
            spect = spect + self.myhparams.augmentation_noise * torch.randn_like(spect)

        return spect

    def get_data(self):
        raise NotImplementedError

    def __getrawitem__(self, idx: int) -> Tuple[Tensor, int, List[int]]:
        rawitem = self.get_data(idx)
        
        spectrogram = self.wave2spect(rawitem["waveform"], rawitem["sample_rate"])

        if spectrogram.size(-1) <= 16:
            spectrogram = torch.cat((int(16. / spectrogram.size(-1)) + 1) * [spectrogram], -1)
    
        if self.mode == "train" or self.myhparams.restrict_spect:
            spectrogram = self.restrict_spect(spectrogram)
        
        if self.mode == "train":
            spectrogram = self.augment_spect(spectrogram)

        rawitem["spectrogram"] = spectrogram
        return rawitem

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, List[int]]:
        item = self.__getrawitem__(idx)
        del item["waveform"], item["sample_rate"]
        return item

class BaseDataModule(pl.LightningDataModule):
    _Dataset_ = None

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.myhparams = SimpleNamespace(**kwargs)
        self.myhparams.data_dir = to_absolute_path(self.myhparams.data_dir)       

    def __repr__(self):
        o = [f"{self.__class__.__name__}"]
        for dataset in ["train", "val", "test"]:
            if hasattr(self, f"{dataset}_dataset"):
                o.append(f'{getattr(self, f"{dataset}_dataset")}\tsize: {len(getattr(self, f"{dataset}_dataset"))}')
        return "\n".join(o)

    def generate_protos(self, cfg):
        data = self.train_dataset if hasattr(self, "train_dataset") else self._Dataset_(self.myhparams, "train")

        proposals = []
        for _ in range(cfg.K+cfg.n_init_proposals):
            spect = data[np.random.randint(len(data))]["spectrogram"]
            if cfg.proto_size_T == 1:
                proposals.append(spect.mean(-1).unsqueeze(-1))
            else:
                start = np.random.randint(spect.size(-1) - cfg.proto_size_T)
                proposals.append(spect[..., start:start+cfg.proto_size_T])

        protos = [proposals[np.random.randint(len(proposals))]]
        for _ in range(cfg.K-1):
            probas = ((torch.vstack(proposals).unsqueeze(1) - torch.vstack(protos).unsqueeze(0))**2).sum(-1).sum(-1).min(-1)[0].numpy()
            probas /= probas.sum()
            protos.append(proposals[np.random.choice(len(proposals), 1, p=probas)[0]])
        
        return protos

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            self.train_dataset = self._Dataset_(self.myhparams, "train")
            self.val_dataset = self._Dataset_(self.myhparams, "valid")
        elif stage in (None, 'validate'):
            self.val_dataset = self._Dataset_(self.myhparams, "valid")

        if stage in (None, 'test'):
            self.test_dataset = self._Dataset_(self.myhparams, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.myhparams.batch_size,
            shuffle=True,
            num_workers=self.myhparams.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.myhparams.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.myhparams.num_workers)

    def show(self, item):

        recovered_wave = self.train_dataset.spect2audio(item["spectrogram"])

        plt.subplot(311)
        plt.plot(item["waveform"][0].numpy())
        plt.subplot(312)
        plt.imshow(item["spectrogram"][0].numpy(), origin='lower', aspect='auto')
        plt.colorbar()
        plt.subplot(313)
        plt.plot(recovered_wave[0].numpy())
        plt.tight_layout()
        plt.show()

        play_audio(item["waveform"], item["sample_rate"])
        play_audio(recovered_wave, item["sample_rate"])