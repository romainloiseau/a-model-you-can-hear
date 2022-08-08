import torch
import torch.nn.functional as F
from matplotlib import cm

class LoggingModel:
    def do_greedy_step(self):
        return (self.current_epoch % int(self.hparams.log_every_n_epochs)) == 0

    @torch.no_grad()
    def greedy_step(self, batch, reconstruction, batch_idx, tag):
        if batch_idx == 0 and self.do_greedy_step():
            if tag == "train":
                self.greedy_model()
                self.greedy_histograms(batch)
            if reconstruction is not None:
                self.greedy_images(batch, reconstruction, tag)

    def greedy_model(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def greedy_histograms(self, batch):        
        self.logger.experiment.add_histogram(
                f"features",
                batch.detach().cpu().flatten(),
                global_step=self.current_epoch
            )

    @torch.no_grad()
    def greedy_images(self, batch, reconstruction, tag):
        n_greedy_images = 4

        # Plot input, output and squared error
        cmap = cm.get_cmap("magma")

        error = (batch - reconstruction).detach().cpu().abs()
        error_toplot = error**2
        error_toplot = error_toplot / error_toplot.view(error.size(0), -1).max(-1)[0].view(-1, 1, 1, 1)

        toplot = [
            torch.from_numpy(cmap(batch.detach().cpu().numpy())[:n_greedy_images, ..., :-1]),
            torch.from_numpy(cmap(reconstruction.detach().cpu().numpy())[:n_greedy_images, ..., :-1]),
            torch.from_numpy(cmap(error_toplot.numpy())[:n_greedy_images, ..., :-1])
        ]
        
        toplot = torch.stack(toplot, 0)[:, :, 0]
        toplot = F.pad(toplot, (0, 0, 1, 1, 1, 1), value=1.)

        
        toplot = toplot.permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2)

        self.logger.experiment.add_image(
            f"{tag}/spect", toplot,
            global_step=self.current_epoch, dataformats='CWH'
        )