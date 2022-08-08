import copy
import hydra
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import time
from .base import DTICallback

from scipy.optimize import linear_sum_assignment

import itertools

class IoU(DTICallback):
    
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        self.confusion_matrix_seq = {}

    def reset(self, tag):
        self.confusion_matrix_seq[tag] = torch.zeros((self.K * self.n_classes, ), dtype=torch.int64)
    
    @torch.profiler.record_function(f"CONFMAT")
    def update_confusion_matrix(self, outputs, batch, tag):
        choice = (outputs["choice"].unsqueeze(-1) == torch.arange(self.K, device=outputs["choice"].device, dtype=outputs["choice"].dtype).unsqueeze(0).unsqueeze(0)).float()
        to_confmat = self.K * batch["label"] + choice.mean(1).argmax(-1)
        unique, counts = torch.unique(to_confmat.flatten(), return_counts=True)
        self.confusion_matrix_seq[tag][unique.detach().cpu().long()] += counts.detach().cpu()

    @torch.no_grad()
    def compute_metrics(self, trainer, pl_module):
        if len(self.confusion_matrix_seq) == 0:
            return
            
        confmat = {}
        for tag in self.confusion_matrix_seq.keys():
            confmat[tag] = self.confusion_matrix_seq[tag].reshape(self.n_classes, self.K).detach().cpu().numpy().astype('float')
        
        if pl_module.hparams.supervised:
            self.best_assign = np.arange(self.K) % self.n_classes   
        elif "train" in confmat.keys():
            self.best_assign = np.argmax(confmat["train"], axis=0)
        else:
            self.best_assign = np.arange(self.K) % self.n_classes  

        for tag, cm in confmat.items():
            confmat_seq = np.vstack([cm[:, self.best_assign == c].sum(axis=1) for c in range(self.n_classes)]).transpose()

            if confmat_seq.sum() != 0:
                confmat_seq /= confmat_seq.sum()
                pl_module.log(f'Acc/OA_{tag}', 100*np.diag(confmat_seq).sum(), on_step=False, on_epoch=True)
                confmat_seq = np.nan_to_num(confmat_seq.astype('float') / confmat_seq.sum(axis=1)[:, np.newaxis]) / self.n_classes
                pl_module.log(f'Acc/AA_{tag}', 100*np.diag(confmat_seq).sum(), on_step=False, on_epoch=True)
                
                trainer.logger.experiment.add_image(
                    f"confmat_seq/{tag}", self.image_confusion_matrix(confmat_seq, pl_module.hparams.data.class_names,
                    ["-".join(np.where(self.best_assign==c)[0].astype(str)) for c in range(self.n_classes)]),
                    global_step=trainer.current_epoch, dataformats='HWC'
                )

        for tag in ["train", "test", "val"]:
            if tag in self.confusion_matrix_seq.keys():
                del self.confusion_matrix_seq[tag]

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        self.update_confusion_matrix(outputs, batch, tag="train")

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.update_confusion_matrix(outputs, batch, tag="val")
    
    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.update_confusion_matrix(outputs, batch, tag="test")

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.reset("val")

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self.reset("test")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.reset("train")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.compute_metrics(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.compute_metrics(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self.compute_metrics(trainer, pl_module)    

    def image_confusion_matrix(self, cm, cm_classes, cm_protoid=None):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """

        cm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

        figure = plt.figure(figsize=(cm.shape[1] / 2.5, cm.shape[0] / 2.5))
        plt.imshow((cm - cm.min()) / (cm.max() - cm.min()), interpolation='nearest', cmap=plt.cm.Blues)
        
        plt.tick_params(labelright=False, right=True, top=True)
        plt.xticks(np.arange(cm.shape[1]), np.arange(cm.shape[1]) if cm_protoid is None else cm_protoid, rotation=90)
        if cm_classes != 0:
            plt.yticks(np.arange(cm.shape[0]), cm_classes, rotation=60)

        threshold = cm.min() + .5 * (cm.max() - cm.min())
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if cm[i, j] != 0:
                color = "white" if cm[i, j] > threshold else "black"
                cmfloat = np.around(100 * cm[i, j], decimals=1 if 100 * cm[i, j] >= 10 else 2)
                plt.text(j, i, cmfloat, horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted prototype')

        s, (width, height) = figure.canvas.print_to_buffer()
        plt.clf()
        plt.close(figure)
        del figure
        return np.fromstring(s, np.uint8).reshape((height, width, 4))