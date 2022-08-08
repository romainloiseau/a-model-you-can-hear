from ast import For
import math
from turtle import forward
import torch
from torch import as_tensor, nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from .base import BaseModel
from tqdm.auto import tqdm
from collections import OrderedDict
from hydra.utils import to_absolute_path
from collections import namedtuple

import torch_scatter

ForwardOut = namedtuple('ForwardOut', ['recerror', 'crossentropy', 'rec', 'choice', 'layer'])

class SpectConv2D(nn.Module):
    def __init__(self, dim_in, dim_out, Fsize, outF, padding_mode):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), padding=(1, 1), padding_mode=padding_mode),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out),
        )
        
        self.collapseF = nn.Sequential(
            nn.Conv2d(dim_out, outF, kernel_size=(Fsize, 1), stride=(1, 1), padding=(0, 0), padding_mode=padding_mode),
            nn.ReLU(),
            nn.BatchNorm2d(outF)
        )

    def forward(self, x):
        return self.collapseF(self.encoder(x)).squeeze(-2)

class DownSpectConv2D(SpectConv2D):

    def __init__(self, dim_in, dim_out, Fsize, factor, padding_mode):
        super().__init__(dim_in, dim_out, Fsize, factor, padding_mode)

        self.down = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), padding_mode=padding_mode),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)
        )

    def forward(self, x):
        residual = self.encoder(x)
        return self.down(residual), self.collapseF(residual).squeeze(-2)

class SpectEncoder2D(nn.Module):
    def __init__(self, start_dim, n_pools, n_mels, outF, padding_mode):
        super().__init__()

        self.encoder = nn.ModuleDict({
            f"down_{i}_{i+1}": DownSpectConv2D(1 if i == 0 else start_dim*2**(i-1), start_dim*2**i, int(n_mels / 2**i), outF, padding_mode) for i in range(n_pools)
        })

        self.last_encoder = SpectConv2D(start_dim*2**(n_pools-1), start_dim*2**n_pools, int(n_mels / 2**n_pools), start_dim*2**(n_pools-1), padding_mode)

        self.n_pools = n_pools        

    def forward(self, x):

        out = {"x": x}

        for i in range(self.n_pools):
            out["x"], out[f"residual_{i}"] = self.encoder[f"down_{i}_{i+1}"](out["x"])

        out["x"] = self.last_encoder(out["x"])

        return out

def UpSpectConv1D(dim_in, dim_out, padding_mode, outF):
    return nn.Sequential(
        nn.Conv1d(dim_in, dim_out, kernel_size=(3), padding=(1), padding_mode=padding_mode),
        nn.ReLU(),
        nn.BatchNorm1d(dim_out),
        nn.ConvTranspose1d(dim_out, dim_out-outF, kernel_size=(2), stride=(2), padding=(0), padding_mode="zeros"),
        nn.ReLU(),
        nn.BatchNorm1d(dim_out-outF)
    )

class SpectDecoder1D(nn.Module):
    def __init__(self, start_dim, n_pools, n_mels, outF, padding_mode, n_parameters):
        super().__init__()

        dim = start_dim*2**(n_pools-1)

        self.decoder = nn.ModuleDict({
            f"up_{i+1}_{i}": UpSpectConv1D(dim, dim, padding_mode, outF) for i in reversed(range(n_pools))
        })

        self.regressor = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=(3), padding=(1), padding_mode=padding_mode),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, n_parameters, kernel_size=(1), stride=(1), padding=(0))
        )
        
        self.n_pools = n_pools

    def forward(self, input):
        for i in reversed(range(self.n_pools)):
            x = torch.cat([self.decoder[f"up_{i+1}_{i}"](x if i != self.n_pools-1 else input["x"]), input[f"residual_{i}"]], 1)
        
        return self.regressor(x)

class AModelYouCanHear(BaseModel):

    def __initmodel__(self, *args, **kwargs):

        self.ce_temperature = nn.Parameter(torch.tensor(self.hparams.temperature_crossentropy))

        self.encoder = SpectEncoder2D(
            self.hparams.start_dim,
            self.hparams.n_pools,
            self.hparams.data.n_mels,
            self.hparams.out_collapseF,
            self.hparams.padding_mode
        )

        self.decoders_n_params = {
            "gain": 1,
            "passfilters": 4,
            "pitch_shifting": 1,
        }
        self.decoders = nn.ModuleDict({
            decoder: SpectDecoder1D(
                self.hparams.start_dim,
                self.hparams.n_pools,
                self.hparams.data.n_mels,
                self.hparams.out_collapseF,
                self.hparams.padding_mode,
                self.decoders_n_params[decoder]*self.hparams.K
            ) for decoder in ["pitch_shifting", "passfilters", "gain"]
        })        

        self.activated_transformations = OrderedDict({
            decoder: True for decoder in ["pitch_shifting", "passfilters", "gain"]
        })
        

        for decoder, n_params in self.decoders_n_params.items():
            self.register_buffer(f"running_mean_{decoder}", torch.zeros((self.hparams.K, n_params)))
        
        if self.hparams.use_bkg:
            raise NotImplementedError

    @torch.no_grad()
    def update_running_stats(self, transformation, choice):
        if np.random.randint(10) == 0:
            for decoder in transformation.keys():
                current = transformation[decoder].flatten(0, -3)
                current = current[torch.arange(current.size(0), device=current.device), choice]
                current = torch_scatter.scatter_mean(current, choice, 0, dim_size=self.hparams.K)

                unqchoice = torch.where(current != 0)[0]

                getattr(self, f"running_mean_{decoder}")[unqchoice] = (1-self.hparams.running_stats_momentum) * getattr(self, f"running_mean_{decoder}")[unqchoice] + self.hparams.running_stats_momentum * current[unqchoice]

    def get_ce_temperature(self):
        return torch.nn.functional.softplus(self.ce_temperature)
            
    def __initprotos__(self, protos):        
        protos = [p[..., 0] for p in protos]
        self._protos = nn.Parameter(torch.stack(
            [p + self.hparams.reassign_noise_scale*torch.randn_like(p) for p in protos], dim=0))

        self.register_buffer("theta", torch.tensor(
            [[1., 0, 0], [0, 1, 0]]).unsqueeze(0))

        self.register_buffer("ftilde",  torch.linspace(0, 1, self.hparams.data.n_mels).unsqueeze(0).unsqueeze(-1))
        self.A = 2595.

        self.register_buffer("arangeF", torch.linspace(0, 1, self.hparams.data.n_mels))

        self.register_buffer("K2choice", torch.arange(self.hparams.K) % self.hparams.data.n_classes)

    def get_protos(self):
        return self._protos

    @torch.no_grad()
    def greedy_model(self):
        self.log(f'ce_temperature', self.get_ce_temperature(), on_step=False, on_epoch=True)

        cmap = cm.get_cmap("magma")

        protos_toplot = self.get_mean_protos_toplot(log=True)

        if protos_toplot.size(-1) < 11:
            protos_toplot = protos_toplot.repeat_interleave(11, dim=-1)

        self.logger.experiment.add_histogram(
            f"proto/_all",
            protos_toplot.detach().cpu().flatten(),
            global_step=self.current_epoch
        )
        for ip, p in enumerate(protos_toplot[:4]):
            self.logger.experiment.add_histogram(
                f"proto/{ip}",
                p.detach().cpu().flatten(),
                global_step=self.current_epoch
            )
        
        protos_toplot = protos_toplot.permute(1, 2, 0, 3)
        protos_toplot = torch.from_numpy(
            cmap(protos_toplot.numpy())[..., :-1])[0]
        protos_toplot = torch.nn.functional.pad(
            protos_toplot, (0, 0, 1, 1), value=1.)
        protos_toplot = protos_toplot.flatten(1, 2)

        self.logger.experiment.add_image(
            f"proto_spect", protos_toplot,
            global_step=self.current_epoch, dataformats='WHC'
        )

    def get_mean_protos_toplot(self, log=False):
        # Plot Protos
        protos_toplot = self.get_protos().unsqueeze(-1)        

        protos_toplot = protos_toplot.detach()

        protos_toplot = protos_toplot.permute(3, 0, 1, 2).unsqueeze(1).unsqueeze(0)
        B, T, L, K, C, F_ = protos_toplot.size()
        N = B*T*K

        running_mean_pitch_shifting = getattr(self, f"running_mean_pitch_shifting").unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat_interleave(T, dim=1)
        running_mean_gain = getattr(self, f"running_mean_gain").unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat_interleave(T, dim=1)
        running_mean_passfilters = getattr(self, f"running_mean_passfilters").unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat_interleave(T, dim=1)

        running_mean_pitch_shifting = running_mean_pitch_shifting.view(-1).contiguous()
        running_mean_gain = running_mean_gain[..., 0].contiguous()

        running_mean_pf0 = running_mean_passfilters[..., 0].contiguous()
        running_mean_pf1 = running_mean_passfilters[..., 1].contiguous()
        running_mean_pf2 = running_mean_passfilters[..., 2].contiguous()
        running_mean_pf3 = running_mean_passfilters[..., 3].contiguous()
        if log:
            self.logger.experiment.add_histogram(
                f"running_mean/pitch_shifting",
                running_mean_pitch_shifting.detach().cpu().flatten(),
                global_step=self.current_epoch
            )
            self.logger.experiment.add_histogram(
                f"running_mean/gain",
                running_mean_gain.detach().cpu().flatten(),
                global_step=self.current_epoch
            )
        running_transforms = (running_mean_pitch_shifting,running_mean_gain,
                                running_mean_pf0, running_mean_pf1, running_mean_pf2, running_mean_pf3)

        protos_toplot = self.do_transforms(protos_toplot.contiguous(), None, 0, "val", B, C,
            self.hparams.data.n_mels, T, K, F_, N, transforms = running_transforms)[0]
        protos_toplot = protos_toplot.flatten(0, 2).permute(1, 2, 3, 0).contiguous().detach().cpu()
        return protos_toplot

    def do_reassignment_k(self, k, newk):
        with torch.no_grad():
            self._protos[k].data.copy_(torch.clone(self._protos[newk].detach(
                )) + self.hparams.reassign_noise_scale*torch.randn_like(self._protos[newk]))
            self._protos[newk].data.copy_(torch.clone(self._protos[newk].detach(
                )) + self.hparams.reassign_noise_scale*torch.randn_like(self._protos[newk]))
            for exp in ["exp_avg", "exp_avg_sq"]:
                self.optimizers().state[self._protos][exp][k].data.copy_(
                    torch.clone(self.optimizers().state[self._protos][exp][newk]))

            if not self.hparams.same_transfo_for_all_protos:

                for decoder in self.activated_transformations.keys():
                    fs = [lambda l: (self.decoders_n_params[decoder] * l + n) for n in range(self.decoders_n_params[decoder])]
                    for param in ["weight", "bias"]:
                        for f in fs:
                            getattr(self.decoders[decoder].regressor[-1], param)[f(k)].data.copy_(
                                torch.clone(getattr(self.decoders[decoder].regressor[-1], param)[f(newk)]))
                            for exp in ["exp_avg", "exp_avg_sq"]:
                                if hasattr(self.optimizers().state[getattr(self.decoders[decoder].regressor[-1], param)], exp):
                                    self.optimizers().state[getattr(self.decoders[decoder].regressor[-1], param)][exp][f(k)].data.copy_(
                                        torch.clone(self.optimizers().state[getattr(self.decoders[decoder].regressor[-1], param)][exp][f(newk)]))

    def get_cross_entropy(self, x):
        return -torch.nn.functional.log_softmax(-self.get_ce_temperature() * x, dim=-1)

    def forward(self, batch, batch_idx=None, tag="train", *args, **kwargs):
        spectrogram = batch["spectrogram"]

        with torch.profiler.record_function(f"FORWARD"):
            protos = self.get_protos()

            B, C, F, T = spectrogram.size()
            K, _, F_ = protos.size()
            N = B*T*K

            with torch.profiler.record_function(f"ID"):
                recs = protos.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
                    B, T, 1, 1, 1, 1)  # K, C, F -> B, T, L, K, C, F

            recs, topad, encoded, transforms = self.do_transforms(recs, spectrogram, batch_idx, tag, B, C, F, T, K, F_, N)

            recs = recs.view(B, T, 1*K, C, F)

            error = ((spectrogram.permute(0, 3, 1, 2).reshape(
                B, T, C, F).unsqueeze(2) - recs)**2).mean(-1).mean(-1)
        
            error = error.reshape(B, T, 1*K, -1).mean(1).reshape(-1, K)

            crossentropy = self.get_cross_entropy(error)

            if self.hparams.supervised and tag=="train":
                batchlabel = batch["label"].flatten()
                isnotlabel = self.K2choice.unsqueeze(0) != batchlabel.unsqueeze(1)
                choice = (error + 100*isnotlabel).argmin(-1)
                recchoice = error.argmin(-1)
            else:
                choice = error.argmin(-1) # Get best proto
                recchoice = choice

            if tag == "train":
                self.update_running_stats(transforms, choice)

            choicearange = torch.arange(choice.size(0), device=choice.device)
            error = error[choicearange, choice]
            crossentropy = crossentropy[choicearange, choice]

            Kchoice = recchoice % K
            Lchoice = torch.div(recchoice, K, rounding_mode="floor")

            rec = recs[choicearange, :, choice].view(
                B, T, C, F).permute(0, 2, 3, 1)

            return ForwardOut(
                error.mean(), crossentropy.mean(),
                rec, Kchoice.view(B, 1).detach(), Lchoice.detach()
            )

    def do_transforms(self, recs, batch, batch_idx, tag, B, C, F, T, K, F_, N, transforms = None):

        if transforms is None:
            encoded, topad = self.encode(2*batch - 1., tag, B, T)
            pitch_shifting, gain, lowpassf0, lowpassG, highpassf0, highpassG, transforms = self.get_transforms(encoded, topad, batch_idx, tag, B, T, K)
        else:
            encoded, topad = None, None
            pitch_shifting, gain, lowpassf0, lowpassG, highpassf0, highpassG = transforms

        recs = self.do_pitch_shifting(recs, pitch_shifting, C, F, F_, N, B, T, K)
        if self.activated_transformations["gain"]:
            recs = self.do_gain(recs, gain)
        if self.activated_transformations["passfilters"]:
            recs = self.do_passfilters(recs, lowpassf0, lowpassG, highpassf0, highpassG)
        return recs, topad, encoded, transforms

    @torch.profiler.record_function("GET T")
    def get_transforms(self, encoded, topad, batch_idx, tag, B, T, K):

        pitch_shifting = self.decoders["pitch_shifting"](encoded)
        gain = self.decoders["gain"](encoded)
        passfilters = self.decoders["passfilters"](encoded)
        
        if tag != "train":
            pitch_shifting = pitch_shifting[..., :topad]
            passfilters = passfilters[..., :topad]
            gain = gain[..., :topad]
        
        pitch_shifting = pitch_shifting.permute(0, 2, 1).reshape(B, T, 1, K).contiguous()# B, L*K*1, T -> B, T, L*K
        gain = gain.permute(0, 2, 1).reshape(B, T, 1, K, self.decoders_n_params["gain"]).contiguous()  # B, K*2, T -> B, T, K*2 -> B, T, K, 2
        passfilters = passfilters.permute(0, 2, 1).reshape(B, T, 1, K, self.decoders_n_params["passfilters"]).contiguous()  # B, K*2, T -> B, T, K*2 -> B, T, K, 2

        if self.hparams.same_transfo_for_all_protos:
            pitch_shifting = pitch_shifting[:, :, :, :1].repeat(1, 1, 1, K)
            gain = gain[:, :, :, :1].repeat(1, 1, 1, K, 1)
            passfilters = passfilters[:, :, :, :1].repeat(1, 1, 1, K, 1)

        pitch_shifting = self.hparams.scale_tanh_pitch_shifting*torch.tanh(pitch_shifting)
        gain = self.hparams.scale_tanh_gain*torch.tanh(gain)
        passfilters = self.hparams.scale_tanh_passfilters*torch.tanh(passfilters)

        transforms = {"pitch_shifting": pitch_shifting.unsqueeze(-1).mean(1).unsqueeze(1), "gain": gain.mean(1).unsqueeze(1), "passfilters": passfilters.mean(1).unsqueeze(1)}

        pitch_shifting = pitch_shifting.view(-1)
        gain = gain[..., 0]

        lowpassf0, lowpassG, highpassf0, highpassG = passfilters[..., 0], passfilters[..., 1], passfilters[..., 2], passfilters[..., 3]

        if batch_idx == 0 and tag == "train":
            self.logger.experiment.add_histogram(
                    f"transfo/pitch_shifting",
                    pitch_shifting.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )
            self.logger.experiment.add_histogram(
                    f"transfo/gain",
                    gain.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )
            self.logger.experiment.add_histogram(
                    f"transfo/lowpassf0",
                    lowpassf0.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )
            self.logger.experiment.add_histogram(
                    f"transfo/lowpassG",
                    lowpassG.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )
            self.logger.experiment.add_histogram(
                    f"transfo/highpassf0",
                    highpassf0.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )
            self.logger.experiment.add_histogram(
                    f"transfo/highpassG",
                    highpassG.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )
                
        return pitch_shifting,gain,lowpassf0, lowpassG, highpassf0, highpassG,transforms
    
    @torch.profiler.record_function(f"ENCODE")
    def encode(self, batch, tag, B, T):
        topad = None

        if tag != "train":
            assert B == 1, "Should have batch size == 1 for validating or testing"
            if T%(2**self.hparams.n_pools) != 0:
                topad =  T%(2**self.hparams.n_pools) - 2**self.hparams.n_pools
            encoded = self.encoder(torch.nn.functional.pad(batch[0], (0, -topad if topad is not None else 0), mode="reflect").unsqueeze(0))
        else:
            encoded = self.encoder(batch)

        return encoded, topad

    def spect2logspect(self, spect):
        return (torch.log10(spect + 10**self.hparams.data.log10_epsilon) - self.hparams.data.log10_epsilon) / self.hparams.data.log10_scale

    def logspect2spect(self, logspect):
        return 10**(self.hparams.data.log10_scale * logspect + self.hparams.data.log10_epsilon) - 10**self.hparams.data.log10_epsilon

    @torch.profiler.record_function(f"PASS")
    def do_passfilters(self, recs, lowpassf0, lowpassG, highpassf0, highpassG):

        highfreq = highpassG.unsqueeze(-1).unsqueeze(-1) * torch.nn.functional.softplus(
            2 * (self.arangeF - .5 + highpassf0.unsqueeze(-1).unsqueeze(-1)),
            beta=10.
        ) #/ (1 + 2*highpassf0.unsqueeze(-1).unsqueeze(-1))

        lowfreq = lowpassG.unsqueeze(-1).unsqueeze(-1) * torch.nn.functional.softplus(
            - 2 * (self.arangeF - .5 + lowpassf0.unsqueeze(-1).unsqueeze(-1)),
            beta=10.
        ) #/ (1 + 2*lowpassf0.unsqueeze(-1).unsqueeze(-1))
        return recs + highfreq + lowfreq
        
    @torch.profiler.record_function(f"AFF")
    def do_gain(self, recs, gain):
        recs = recs + gain.unsqueeze(-1).unsqueeze(-1) 
        return recs

    @torch.profiler.record_function(f"PITCH")
    def do_pitch_shifting(self, recs, pitch_shifting, C, F, F_, N, B, T, K):

        if self.activated_transformations["pitch_shifting"]:
            pitch_shifting = torch.exp((pitch_shifting.view(B, T, 1, K)).view(-1))
        else:
            pitch_shifting = torch.exp((0*pitch_shifting.detach().view(B, T, 1, K)).view(-1))
            
        grid = self.A * torch.log10(1 + pitch_shifting.unsqueeze(-1).unsqueeze(-1) * (10**(self.ftilde / self.A) - 1))
        grid = 2 * grid - 1
        grid = torch.stack([0*grid.detach(), grid], -1)

        recs = recs.view(-1, C, F_, 1)
        recs = torch.nn.functional.grid_sample(
                    recs,
                    grid,
                    mode='bilinear',
                    padding_mode='border', #border, zeros ?
                    align_corners=False)
            
        return recs.view(B, T, 1, K, C, F)