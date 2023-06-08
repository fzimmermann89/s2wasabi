from nets import *
from unet import *
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial


class unetonly3_lessFIXT2(nn.Module):
    def __init__(self, xm, xs, ym, ys):
        super().__init__()
        self.UNet = Unet(
            dim=2,
            channels_in=31,
            channels_out=1,
            layer=3,
            filters=64,
            conv_per_enc_block=2,
            conv_per_dec_block=2,
            norm=False,
            bias=True,
            up_mode="linear",
            activation=partial(nn.LeakyReLU, inplace=True),
            feature_growth=lambda depth: [2, 1.5, 1, 1, 1, 1][depth],
        )
        features_out = self.UNet.last[0].in_channels
        self.UNet.last = nn.Identity()
        self.out = nn.Conv2d(features_out, 6, 1)
        self.fe = nn.Sequential(
            nn.Conv2d(features_out, 32, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.register_buffer("xm", torch.as_tensor(xm))
        self.register_buffer("xs", torch.as_tensor(xs))
        self.register_buffer("ym", torch.as_tensor(ym))
        self.register_buffer("ys", torch.as_tensor(ys))

    def denorm(self, xc, xf):
        ret = torch.cat(
            [
                F.softplus(xc[:, 0:1] * self.xs[:, 0:1] + self.xm[:, 0:1, ...], beta=20)
                + 1e-8,
                (xc[:, 1:2]) * self.xs[:, 2:3] + self.xm[:, 2:3, ...],
                F.softplus(
                    xc[:, 2:3] * (self.xs[:, 3:4]) + self.xm[:, 3:4, ...] + 1e-8, beta=5
                ),
                F.softplus(xc[:, 3:6], beta=5) * self.xs[:, (0, 2, 3)] + 1e-8,
                F.softplus(xf, beta=5) * 0.1 + 1e-8,
            ],
            1,
        )
        return ret

    def forward(self, x, ret_FCN=False):
        xin = (x - self.ym) / self.ys
        xu = self.UNet(xin)
        xout = self.out(xu)
        xf = self.fe(xu.detach())

        return self.denorm(xout, xf)


class pixelwise(torch.nn.Module):
    hiddendims = (128, 128, 256, 256, 128, 128)

    def __init__(self, xm, xs, ym, ys):
        super().__init__()
        self.net = FCNet(
            input_dim=31,
            hidden_dims=self.hiddendims,
            activation=partial(torch.nn.ELU, inplace=True),
            output_dim=6,
        )
        self.register_buffer("xm", torch.as_tensor(xm))
        self.register_buffer("xs", torch.as_tensor(xs))
        self.register_buffer("ym", torch.as_tensor(ym))
        self.register_buffer("ys", torch.as_tensor(ys))

    def denorm(self, xc):
        ret = torch.cat(
            [
                F.softplus(xc[:, 0:1] * self.xs[:, 0:1] + self.xm[:, 0:1, ...], beta=20)
                + 1e-8,
                (xc[:, 1:2]) * self.xs[:, 2:3] + self.xm[:, 2:3, ...],
                F.softplus(
                    xc[:, 2:3] * (self.xs[:, 3:4]) + self.xm[:, 3:4, ...] + 1e-8, beta=5
                ),
                F.softplus(xc[:, 3:6], beta=5) * self.xs[:, (0, 2, 3)] + 1e-8,
            ],
            1,
        )
        return ret

    def norm(self, x):
        ret = (x - self.ym) / self.ys
        return ret

    def forward(self, x):
        x = self.norm(x)
        x = self.net(x)
        x = self.denorm(x)
        return x
