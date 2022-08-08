import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from .base import DTICallback
from types import SimpleNamespace

class Reassigner(DTICallback):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.threshold = self.myhparams.threshold
        self.wait_before_start = self.myhparams.wait_before_start

    def on_train_epoch_start(self, *args, **kwargs):
        self.count_assignments = torch.zeros((self.K, ))

    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs) -> None:
        with torch.no_grad():
            unique, counts = torch.unique(outputs["choice"].flatten(), return_counts=True)
            self.count_assignments[unique.detach().cpu()] += counts.detach().cpu()

    def on_train_epoch_end(self, trainer, pl_module):
        count_assignments = self.K * self.count_assignments / self.count_assignments.sum()

        if self.do_greedy_step(trainer.current_epoch):
            figure = plt.figure()
            bins = np.arange(count_assignments.size(0) + 1) - .5
            centroids = (bins[1:] + bins[:-1]) / 2
            plt.hist(centroids, bins=bins,
                weights=count_assignments.numpy())
            plt.plot(bins, 1+0*bins, color="black")
            plt.plot(bins, self.threshold+0*bins, color="red")
            plt.ylim((0.01, 10))
            plt.xlim((-.5, self.K-.5))
            plt.yscale("log")
            plt.xlabel("Prototype")
            plt.ylabel("Proportion *K")
            s, (width, height) = figure.canvas.print_to_buffer()
            plt.clf()
            plt.close(figure)
            del figure
            trainer.logger.experiment.add_image(
                f"assignments", np.fromstring(s, np.uint8).reshape((height, width, 4)),
                global_step=trainer.current_epoch, dataformats='HWC'
            )
        
        if not self.supervised and trainer.current_epoch >= self.wait_before_start:
            reassign = count_assignments < self.threshold
            reassign[-1] = False

            reassignement_proba = count_assignments * ~reassign / (count_assignments * ~reassign).sum()
            reassignemnt = np.random.choice(count_assignments.size(0), size=count_assignments.size(0), p=reassignement_proba.numpy())

            for k, do_reassign in enumerate(reassign):
                if do_reassign:
                    self.send_message(f"Reassignement {k} <- {reassignemnt[k]}", trainer.current_epoch)
                    pl_module.do_reassignment_k(k, reassignemnt[k])