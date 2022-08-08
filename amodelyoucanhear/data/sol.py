import os
import os.path as osp
from tqdm.auto import tqdm
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
from .base import BaseDataModule, BaseDataset

class SOLDataset(BaseDataset):

    def __init__(self, myhparams, mode, *args, **kwargs):
        super().__init__(myhparams, mode, *args, **kwargs)

        if not osp.exists(to_absolute_path("splits")):
            os.mkdir(to_absolute_path("splits"))
        if not osp.exists(to_absolute_path(osp.join("splits", "sol"))):
            os.mkdir(to_absolute_path(osp.join("splits", "sol")))

        if not osp.exists(to_absolute_path(osp.join("splits", "sol", f"{mode}_split.csv"))):
        
            wav_path = []
            gts = []

            np.random.seed(0)
            for i, c in enumerate(self.myhparams.class_names):
                if osp.isdir(osp.join(self.myhparams.data_dir, c)):
                    for style in os.listdir(osp.join(self.myhparams.data_dir, c)):
                        if osp.isdir(osp.join(self.myhparams.data_dir, c, style)):
                            for file in os.listdir(osp.join(self.myhparams.data_dir, c, style)):
                                if osp.splitext(file)[-1] == ".wav":
                                    p = np.random.uniform(0, 1)

                                    sel_train = mode == "train" and p >= 3*myhparams.prop_val
                                    sel_val = mode == "valid" and p < myhparams.prop_val
                                    sel_test = mode == "test" and p >= myhparams.prop_val and p < 3*myhparams.prop_val
                                    if sel_train or sel_val or sel_test:
                                        wav_path.append(osp.join(c, style, file))
                                        gts.append(i)

            ab = np.zeros(len(wav_path), dtype=[('label', int), ('path', 'U100')])
            ab['label'] = np.array(gts)
            ab['path'] = np.array(wav_path)

            np.savetxt(
                to_absolute_path(osp.join("splits", "sol", f"{mode}_split.csv")), ab,
                fmt="%i,%s"
            )

        ab = np.loadtxt(
            to_absolute_path(osp.join("splits", "sol", f"{mode}_split.csv")),
            delimiter=",", dtype=[('label', int), ('path', 'U100')], comments="    "
        )
        self.wav_path = ab["path"]
        self.gts = ab['label']
        
        if self.myhparams.load_in_memory:
            self.wav = [self.load(idx) for idx in tqdm(range(len(self)), desc=f"Loading {self.mode} dataset")]

    def __len__(self):
        return len(self.wav_path)

    def get_data(self, idx):
        if self.myhparams.load_in_memory:
            waveform, sample_rate = self.wav[idx]
        else:
            waveform, sample_rate = self.load(idx)
        assert sample_rate == 44100

        waveform = waveform.mean(0).unsqueeze(0)
        
        return {"waveform": waveform, "sample_rate": sample_rate, "label": self.gts[idx]}

    def load(self, idx):
        return torchaudio.load(osp.join(self.myhparams.data_dir, self.wav_path[idx]))

class SOLDataModule(BaseDataModule):
    _Dataset_ = SOLDataset