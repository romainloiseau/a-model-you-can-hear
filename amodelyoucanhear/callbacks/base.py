from tqdm.auto import tqdm

from types import SimpleNamespace
from pytorch_lightning.callbacks import Callback

class DTICallback(Callback):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.myhparams = SimpleNamespace(**kwargs)

    def send_message(self, message, epoch=0):
        tqdm.write(f"Epoch {epoch}:\t{message}")

    def do_greedy_step(self, epoch):
        return (epoch % int(self.log_every_n_epochs)) == 0

    def on_train_start(self, trainer, pl_module):
        self.K = pl_module.hparams.K
        self.log_every_n_epochs = pl_module.hparams.log_every_n_epochs

        self.n_classes = pl_module.hparams.data.n_classes

        self.supervised = pl_module.hparams.supervised

    def on_test_start(self, trainer, pl_module):
        self.K = pl_module.hparams.K
        self.log_every_n_epochs = pl_module.hparams.log_every_n_epochs

        self.n_classes = pl_module.hparams.data.n_classes