import torchaudio
import numpy as np
from tqdm.auto import tqdm
from hydra.utils import to_absolute_path
from collections import defaultdict
import os
import os.path as osp

import copy

from .base import BaseDataModule, BaseDataset

class LibriSpeechDataset(BaseDataset):

    def __init__(self, myhparams, mode, *args, **kwargs):
        super().__init__(myhparams, mode, *args, **kwargs)
        self.data = torchaudio.datasets.LIBRISPEECH(myhparams.data_dir, download=True, url=myhparams.url)

        if not osp.exists(to_absolute_path("splits")):
            os.mkdir(to_absolute_path("splits"))
        if not osp.exists(to_absolute_path(osp.join("splits", "librispeech"))):
            os.mkdir(to_absolute_path(osp.join("splits", "librispeech")))

        if not osp.exists(to_absolute_path(osp.join("splits", "librispeech", f"{mode}_{myhparams.url}_K{myhparams.n_classes}_split.csv"))):
            if not osp.exists(to_absolute_path(osp.join("splits", "librispeech", f"all_ids_{myhparams.url}.npy"))):
                np.save(
                    to_absolute_path(osp.join("splits", "librispeech", f"all_ids_{myhparams.url}.npy")),
                    np.array([[i, d[0].size(-1), d[3]] for i, d in enumerate(self.data)]),
                )

            all_ids = np.load(
                    to_absolute_path(f"splits/librispeech/all_ids_{myhparams.url}.npy")
                )

            id2len = defaultdict(int)
            for id in all_ids:
                id2len[id[2]] += id[1] #1

            keep_ids = sorted(id2len, key=id2len.get, reverse=True)[:myhparams.n_classes]
            id2label = {id: label for label, id in enumerate(keep_ids)}

            np.random.seed(0)
            newwalker = []
            gts = []
            for id in all_ids:
                if id[2] in keep_ids:
                    p = np.random.uniform(0, 1)
                    sel_train = mode == "train" and p >= 3*myhparams.prop_val
                    sel_val = mode == "valid" and p < myhparams.prop_val
                    sel_test = mode == "test" and p >= myhparams.prop_val and p < 3*myhparams.prop_val
                    if sel_train or sel_val or sel_test:
                        newwalker.append(self.data._walker[int(id[0])])
                        gts.append(id2label[id[2]])

            ab = np.zeros(len(newwalker), dtype=[('label', int), ('path', 'U100')])
            ab['label'] = np.array(gts)
            ab['path'] = np.array(newwalker)

            np.savetxt(
                to_absolute_path(osp.join("splits", "librispeech", f"{mode}_{myhparams.url}_K{myhparams.n_classes}_split.csv")), ab,
                fmt="%i,%s"
            )

        ab = np.loadtxt(
            to_absolute_path(osp.join("splits", "librispeech", f"{mode}_{myhparams.url}_K{myhparams.n_classes}_split.csv")),
            delimiter=",", dtype=[('label', int), ('path', 'U100')], comments="    "
        )
        self.data._walker = ab["path"]
        self.gts = ab['label']

    def get_data(self, idx):
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.data[idx]
        assert sample_rate == 16000
        return {"waveform": waveform, "sample_rate": sample_rate, "label": self.gts[idx]}

class LibriSpeechDataModule(BaseDataModule):
    _Dataset_ = LibriSpeechDataset