import torch
import torch.nn as nn

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
        self.upsample1 = nn.Upsample(scale_factor=tuple(pool_size))

        conv2 = nn.Conv2d(n_filters[0], n_filters[1], kernel_size=kernel_size, padding=(2, 2), padding_mode=padding_mode, dilation=dilation)
        act2 = nn.LeakyReLU()
        self.conv2 = nn.Sequential(*[conv2, act2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=tuple(pool_size))
        self.upsample2 = nn.Upsample(scale_factor=tuple(pool_size))

        conv3 = nn.Conv2d(n_filters[1], n_filters[2], kernel_size=kernel_size, padding=(2, 2), padding_mode=padding_mode, dilation=dilation)
        act3 = nn.LeakyReLU()
        self.conv3 = nn.Sequential(*[conv3, act3])

    def forward(self, x):
        y = self.conv1(x)
        y_pool = self.maxpool1(y)
        y_up = self.upsample1(y_pool)

        mask1 = (y >= y_up).float()

        y = self.conv2(y_pool)
        y_pool = self.maxpool2(y)
        y_up = self.upsample2(y_pool)
        mask2 = (y >= y_up).float()

        y = self.conv3(y_pool)
        return y, mask1, mask2

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

        conv1 = nn.ConvTranspose2d(n_filters[2], n_filters[1], kernel_size=kernel_size, padding=(2, 2), padding_mode="zeros", dilation=dilation)
        upsample1 = nn.Upsample(scale_factor=tuple(pool_size))
        self.deconv1 = nn.Sequential(*[conv1, upsample1])

        conv2 = nn.ConvTranspose2d( n_filters[1], n_filters[0], kernel_size=kernel_size, padding=(2, 2), padding_mode="zeros", dilation=dilation)
        upsample2 = nn.Upsample(scale_factor=tuple(pool_size))
        self.deconv2 = nn.Sequential(*[conv2, upsample2])

        conv3 = nn.ConvTranspose2d(n_filters[0], start_dim, kernel_size=kernel_size, padding=(2, 2), padding_mode="zeros", dilation=dilation)
        act3 = nn.Tanh()
        self.deconv3 = nn.Sequential(*[conv3, act3])

    def forward(self, x, mask1, mask2):
        deconv = self.act(mask2 * self.deconv1(x))
        deconv = self.act(mask1 * self.deconv2(deconv))
        deconv = self.deconv3(deconv)
        return deconv


def custom_loss(predictions, labels, reconstruction, input, distances):
    loss_ce = nn.CrossEntropyLoss()
    loss_rec = lambda inp, out: torch.mean(torch.sum(torch.pow(inp - out, 2), dim=(1, 2, 3)))
    loss_proto = lambda distance: torch.min(distance, dim=0)[0].mean() + torch.min(distance, dim=1)[0].mean()
    return loss_ce(predictions, labels), loss_proto(distances), loss_rec(input, reconstruction)


class APNet(BaseModel):
    def __initmetrics__(self):
        self.loss = custom_loss

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

        self.T = self.hparams.size_T // 4
        self.F = self.hparams.size_F // 4
        self.M = self.hparams.M
        self.Cfeat = self.hparams.n_filters[-1]

        self.prototype_layer = torch.nn.Parameter(nn.init.uniform_(torch.empty((self.M,
                                                                                self.Cfeat,
                                                                                self.F,
                                                                                self.T)),
                                                                   a=0.0, b=1.0
                                                                   ))  # is K*Cfeat*F*T

        self.frequency_weights = torch.nn.Parameter(torch.zeros((self.M, self.F)))  # is KxF

        self.linear = nn.Linear(self.M, self.hparams.K, bias=False)

    def greedy_model(self):
        return

    def get_frequency_weights(self):
        return torch.softmax(self.frequency_weights, dim=1)

    @torch.profiler.record_function(f"FORWARD")
    def forward(self, batch, *args, **kwargs):
        latent_features, mask1, mask2 = self.encoder(2*batch["spectrogram"]-1)
        reconstructed_input = self.decoder(latent_features, mask1, mask2)
        distance = torch.sum(torch.pow((latent_features[:, None, ...].expand(-1, self.M, -1, -1, -1) - self.prototype_layer), 2), dim=(2, 4))
        similarity = torch.exp(-distance)  # B*K*F
        mean = torch.sum(similarity*self.get_frequency_weights(), dim=2)
        prediction = self.linear(mean)
        return prediction, reconstructed_input, distance.mean(-1), mean

    def global_step(self, batch, batch_idx, tag):
        predictions, reconstruction, distances, _ = self.forward(batch, batch_idx, tag)

        with torch.profiler.record_function(f"LOSS"):
            loss_ce, loss_proto, loss_rec = self.loss(predictions, batch['label'], reconstruction, 2*batch["spectrogram"]-1, distances)

            loss = self.hparams.alpha * loss_ce + self.hparams.beta * loss_proto + self.hparams.gamma * loss_rec

        with torch.no_grad():
            with torch.profiler.record_function(f"LOGGERS"):
                self.log(f'Loss/{tag}',
                         loss, on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))

                self.log(f'Losses/proto/{tag}', loss_proto / (self.T * self.F), on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))
                self.log(f'Losses/rec/{tag}', loss_rec / (4 * self.hparams.size_T * self.hparams.size_F), on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))

                self.log(f'Losses/ce/{tag}', loss_ce, on_step=False, on_epoch=True, batch_size=batch["spectrogram"].size(0))

                self.greedy_step(batch["spectrogram"], .5 + reconstruction/2., batch_idx, tag)

        return {"loss": loss, "choice": predictions.argmax(-1).unsqueeze(-1)}

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optim._target_)(
            self.parameters(),
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay
        )

        return {
            "optimizer": optimizer,
        }