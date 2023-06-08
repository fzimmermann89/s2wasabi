import torch
import numpy as np
from functools import partial, reduce
from pathlib import Path
import multiprocessing
import scipy.ndimage as snd
from torch.utils.data import DataLoader
from torchvision import transforms as T
import kornia
from torch import Tensor
from typing import *

from datasets import (
    BrainwebR1R2ClassesSlices,
    Hdf5DataSet,
    WasabiDS,
)
from util import (
    AddFillMask,
    RemoveFillMask,
    cutnan,
    random_gaussians,
    random_poly2d,
    wif,
)
from wasabifw import WasabiMzAna as fwfunction
from augments import correlatednoise, polynoise, randmult


class RandomPoly(kornia.augmentation.IntensityAugmentationBase2D):
    def __init__(
        self,
        scales=(1.0, 1.0),
        same_on_batch=False,
        p=1.0,
        keepdim=False,
        mode="add",
        return_transform=None,
    ):
        super().__init__(
            p=p,
            return_transform=return_transform,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
        )
        samplers = [
            ((-1 / 2 * s, 1 / 2 * s), f"xscale{i}", None, None)
            for i, s in enumerate(scales)
        ] + [
            ((-1 / 2 * s, 1 / 2 * s), f"yscale{i}", None, None)
            for i, s in enumerate(scales)
        ]
        self._param_generator = kornia.augmentation.random_generator.PlainUniformGenerator(
            *samplers
        )
        self.order = len(scales)
        self.mode = mode

    def apply_transform(self, input, params, flags, transform=None):
        xscales = [params[f"xscale{i}"].to(input) for i in range(self.order)]
        yscales = [params[f"yscale{i}"].to(input) for i in range(self.order)]
        x = torch.linspace(-1, 1, input.shape[-2])[None, :].to(input)
        y = torch.linspace(-1, 1, input.shape[-1])[None, :].to(input)
        polyx = sum(xscales[i][:, None] * x ** (i + 1) for i in range(self.order)) + 1
        polyy = sum(yscales[i][:, None] * y ** (i + 1) for i in range(self.order)) + 1
        poly = polyx[..., None, :, None] * polyy[..., None, None, :]
        if self.mode == "add":
            return input + (poly - 1)
        if self.mode == "mul":
            return input * poly


class Clamp(kornia.augmentation.IntensityAugmentationBase2D):
    def __init__(self, range=(1e-6, 1e6)):
        super().__init__(p=1.0, same_on_batch=True, keepdim=True)
        self.range = range

    def apply_transform(self, input, *args, **kwargs):
        return torch.clamp(input, min=self.range[0], max=self.range[1])


class R2_from_R1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.poly = RandomPoly((0.05, 0.1, 0.05), mode="mul")

    def forward(self, x):
        x = 1000 / x
        x = 10 + x / 8 + torch.sigmoid(x / 40 - 50) * (2000 - x / 15)
        x = 1000 / x
        x = self.poly(x).reshape(x.shape)
        return x


def randomrot90(*args, p=0.5):
    if torch.rand(1) >= p:
        return args
    else:
        return tuple([torch.rot90(a, 1, (-1, -2)) for a in args])


class RandomMaskCrop:
    def __init__(self, sizex, sizey):
        self.size = (sizex, sizey)

    def __call__(self, img, mask):
        ind = torch.all(mask == 0, -1)
        x0, x1 = np.argmin(ind), np.argmin(ind.flip(-1))
        if mask.shape[-2] - x0 - x1 > self.size[0]:
            x = torch.randint(int(x0), int(mask.shape[-2] - x1 - self.size[0]), (1,))
        else:
            x = max(0, mask.shape[-1] // 2 - self.size[0] // 2)
        x0, x1 = x, min(x + self.size[0], mask.shape[-2])

        ind = torch.all(mask == 0, -2)
        y0, y1 = np.argmin(ind), np.argmin(ind.flip(-1))
        if mask.shape[-1] - y0 - y1 > self.size[1]:
            y = torch.randint(int(y0), int(mask.shape[-1] - y1 - self.size[1]), (1,))
        else:
            y = max(0, mask.shape[-1] // 2 - self.size[1] // 2)
        y0, y1 = y, min(y + self.size[1], mask.shape[-1])

        mask = mask[..., x0:x1, y0:y1]
        img = img[..., x0:x1, y0:y1]
        return img, mask


def center(img, mask):
    x, y = torch.meshgrid(
        *[torch.linspace(-s / 2, s / 2 - 1, s) for s in mask.shape[-2:]]
    )
    shifts = -int(x[mask].mean()), -int(y[mask].mean())
    mask = torch.roll(mask, shifts, (-2, -1))
    img = torch.roll(img, shifts, (-2, -1))
    return img, mask


def RicianNoise(x, sigma):
    e1, e2 = torch.randn(2, *x.shape) * sigma
    ret = ((e1 + x).square() + e2.square()).sqrt()
    return ret


def gamma_rician_noise(
    image: Tensor,
    mask: Tensor,
    meanvar: float = 0.7e-2,
    varvar: float = 2.1e-4,
    same_axes: Tuple[Tuple[int]] = ((-1, -2, -3),),
):
    m = torch.as_tensor(meanvar, dtype=torch.float64)
    v = torch.as_tensor(varvar, dtype=torch.float64) * len(same_axes)
    d = torch.distributions.gamma.Gamma(m ** 2 / v, m / v)
    var = []
    for sa in same_axes:
        s = torch.tensor(image.shape)
        s[(sa,)] = 1
        mean = torch.sum(image, dim=sa, keepdim=True) / torch.sum(
            mask, dim=sa, keepdim=True
        )
        var.append(d.sample(s) / mean ** 2)
    var = sum(var) / len(var)
    sigma = (mask > 0) * torch.sqrt(var)
    noisy = torch.nan_to_num(RicianNoise(image, sigma), 0, 0, 0)
    return noisy, sigma


class PseudoLabelPermutation(torch.utils.data.Dataset):
    def __init__(self, labels, masks, transforms=None):
        self.n, self.channels = labels.shape[:2]
        self.labels = []
        self.masks = []
        for img, mask in zip(labels, masks):
            # img, mask = center(img, mask)
            self.labels.append(img)
            self.masks.append(mask)
        self.transforms = transforms

    def __len__(self):
        return self.n ** self.channels

    def __getitem__(self, i):
        if i >= len(self):
            raise ValueError(
                f"{i} is out of bounds for {self.n} sets of {self.channels} channels = {len(self)} possible combinations"
            )
        indices = np.unravel_index(i, self.channels * (self.n,))
        labels = [self.labels[index][i] for i, index in enumerate(indices)]
        masks = [self.masks[index] for index in indices]
        if self.transforms is not None:
            labels, mask = zip(
                *(T(l, m.float()) for l, m, T in zip(labels, masks, self.transforms))
            )
        labels = torch.stack([l.squeeze() for l in labels], 0)
        mask = torch.all(torch.stack([m.bool().squeeze() for m in mask], 0), 0)
        return labels, mask.bool()


class WasabiPseudoLabels(torch.utils.data.Dataset):
    def __init__(self, labels, masks, fw, size=(128, 128)):
        labels = labels.moveaxis(1, 0)
        labels[:, ~masks] = torch.tensor([0.8, 0.0, 3.75])[:, None]
        labels = labels.moveaxis(0, 1)
        self.T = (
            kornia.augmentation.container.AugmentationSequential(
                RandomPoly((0.1, 0.2, 0.1), mode="mul"),
                kornia.augmentation.RandomHorizontalFlip(),
                kornia.augmentation.RandomVerticalFlip(),
                kornia.augmentation.RandomAffine(5, 0.15, (0.9, 1.1), 5, p=0.8),
                Clamp((1e-5, 1e3)),
                data_keys=["input", "mask",],
            ),
            kornia.augmentation.container.AugmentationSequential(
                RandomPoly((0.1, 0.2)),
                kornia.augmentation.RandomHorizontalFlip(),
                kornia.augmentation.RandomVerticalFlip(),
                kornia.augmentation.RandomAffine(5, 0.05, (0.9, 3.0), 5, p=0.9),
                kornia.augmentation.RandomGaussianBlur((7, 7), (1.0, 5.0),),
                Clamp((-4, 4)),
                data_keys=["input", "mask",],
            ),
            kornia.augmentation.container.AugmentationSequential(
                RandomPoly((0.1, 0.2)),
                kornia.augmentation.RandomHorizontalFlip(),
                kornia.augmentation.RandomVerticalFlip(),
                kornia.augmentation.RandomAffine(5, 0.1, (0.9, 3.0), 5, p=0.9),
                kornia.augmentation.RandomGaussianBlur((7, 7), (2.0, 5.0),),
                Clamp((0.1, 5)),
                data_keys=["input", "mask",],
            ),
        )

        self.ds = PseudoLabelPermutation(labels, masks, transforms=self.T)
        self.fw = fw
        self._maskvaluesX = torch.tensor([0.8, 10, 0.0, 3.75])[:, None]
        self._maskvaluesY = self.fw(self._maskvaluesX).squeeze()[:, None]
        self.R2_from_R1 = R2_from_R1()
        self.crop = RandomMaskCrop(*size)
        self.sel = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])

    def __len__(self):
        return len(self.ds)

    def update(self, labels, mask):
        self.ds = PseudoLabelPermutation(labels, mask, transforms=self.T)

    def __getitem__(self, n):
        x, mask = self.ds[n]
        x, mask = randomrot90(x, mask)
        x, mask = self.crop(x, mask)
        x0 = torch.nan_to_num_(
            torch.clamp_(x[0], 1e-1, 20), nan=float(self._maskvaluesX[0])
        )
        x1 = torch.nan_to_num_(
            torch.clamp_(self.R2_from_R1(x[0]), 1e-1, 40),
            nan=float(self._maskvaluesX[1]),
        )
        x2 = torch.nan_to_num_(
            torch.clamp_(x[1], -2, 2), nan=float(self._maskvaluesX[2])
        )
        x3 = torch.nan_to_num_(
            torch.clamp_(x[2], 0.25 * 3.75, 4 * 3.75), nan=float(self._maskvaluesX[3])
        )
        x = torch.stack((x0, x1, x2, x3), 0)
        y = self.fw(x.unsqueeze(1)).squeeze(0).float()
        mask = torch.logical_and(
            torch.all(torch.isfinite(y), axis=0, keepdim=True), torch.as_tensor(mask)
        )
        y = y.nan_to_num_().clamp_(0, 2)

        mask = torch.as_tensor(~snd.binary_dilation(~mask, self.sel))

        x[:, ~mask[0]] = self._maskvaluesX
        y[:, ~mask[0]] = self._maskvaluesY
        SNR = 5 + 30 * torch.rand(1)
        sigma = (torch.sum(y * mask) / (mask.sum() * y.shape[0])) / SNR
        yn = RicianNoise(y, sigma)
        yn = torch.clamp_(torch.abs_(yn), min=1e-10, max=5)

        return yn.float(), mask.float(), x.float()


class WasabiSynthetic(torch.utils.data.Dataset):
    def __init__(self, ds):
        T = None
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, n):
        items = self.ds[n]
        x, y, yn, mask, *_ = items
        return yn, mask, x


class WasabiSelf(torch.utils.data.Dataset):
    def __init__(self, data, masks, repeats=1, size=(128, 128)):
        self.T = kornia.augmentation.container.AugmentationSequential(
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomVerticalFlip(),
            Clamp((-1, 5)),
            data_keys=["input", "mask",],
        )
        self.data = data.float()
        self.masks = masks.float()
        self.repeats = repeats
        self.crop = RandomMaskCrop(*size)

    def __len__(self):
        return len(self.data) * self.repeats

    def __getitem__(self, n):
        n = n % len(self.data)
        x, m = self.data[n], self.masks[n]
        x, m = self.T(x, m)
        x, m = randomrot90(x, m)
        x.squeeze_(0), m.squeeze_(0)
        x, m = self.crop(x, m)
        return x, m


class FWFakeR2(torch.nn.Module):
    def __init__(self, fwfunction):
        super().__init__()
        self.fwfunction = fwfunction

    def R2_from_R1(self, x):
        x = 1000 / x
        x = 10 + x / 8 + torch.sigmoid(x / 40 - 50) * (2000 - x / 15)
        x = 1000 / x
        return x

    def forward(self, x):
        r1 = torch.clamp(x[:, 0], min=-1e3, max=1e3)
        r2 = torch.clamp(self.R2_from_R1(r1.detach()), min=-1e3, max=1e3)
        b0 = torch.clamp(x[:, 1], min=-1e3, max=1e3)
        b1 = torch.clamp(x[:, 2], min=-1e3, max=1e3)
        return self.fwfunction((r1, r2, b0, b1))


def getLoaderFineTune(path, selfdata, selfmasks, pseudolabels, pseudomask):
    maxthreads = multiprocessing.cpu_count()
    path = Path(path)
    size = (128, 128)

    offset = np.linspace(-2, 2, 31).astype(np.float32)
    trec = np.array(
        [0.5, 1, 1.5, 2, 2.5, 3] + (len(offset) - 12) * [1.5] + [3, 2.5, 2, 1.5, 1, 0.5]
    ).astype(np.float32)
    fwdata = torch.jit.script(fwfunction(offset, trec + 0.0055, 42.5764 * 3)).cpu()

    validation_path = path / "validation5.hdf5"
    _field_blur = T.GaussianBlur(7, sigma=(0.2, 1.0))
    field_blur = lambda x: _field_blur(x[None, None, ...])[0, 0]

    transformsSyn = (
        AddFillMask,
        T.GaussianBlur(7, sigma=(0.01, 1.5)),
        T.RandomAffine(
            10,
            scale=(0.35, 0.55),
            fill=0,
            shear=10,
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        RemoveFillMask,
        cutnan,
        T.RandomCrop(size, pad_if_needed=True, fill=np.nan),
        T.RandomApply([partial(torch.rot90, dims=(-1, -2))]),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        partial(randmult, sigma=0.2),
        T.RandomChoice(
            [
                partial(correlatednoise, scale=(8, 32), **params)
                for params in [
                    dict(strength=(0.04, 0.07), mode="add"),
                    dict(strength=(0.02, 0.02), mode="mul"),
                    dict(strength=(0.075, 0.015), mode="invadd"),
                ]
            ]
        ),
        T.RandomApply(
            [
                partial(
                    reduce,
                    lambda r, params: polynoise(r, **params),
                    [
                        dict(strength=(0.25, 0.3), mode="add"),
                        dict(strength=(0.06, 0.06), mode="mul"),
                        dict(strength=(0.35, 0.075), mode="invadd"),
                    ],
                )
            ],
            p=0.9,
        ),
        T.GaussianBlur(3, sigma=(0.01, 0.2)),
    )

    dlSyn = DataLoader(
        WasabiSynthetic(
            WasabiDS(
                BrainwebR1R2ClassesSlices(
                    path / "train",
                    axis=0,
                    cuts=(96, 64, 16, 16, 16, 16),
                    transforms=transformsSyn,
                ),
                (
                    lambda *s: field_blur(
                        correlatednoise(
                            random_poly2d(*s, strength=(1.3, 0.008), p=0.9, sall=0.1),
                            scale=[12, 32],
                            strength=0.05,
                        )
                    )
                ),
                (
                    lambda *s: field_blur(
                        correlatednoise(
                            random_gaussians(
                                *s, s=0.1, scales=(0.3, 0.6, 1.0), p=0.9, sall=0.1
                            ),
                            scale=[24, 64],
                            strength=0.01,
                        )
                    )
                ),
                fwdata.cpu(),
                gamma_rician_noise,
                return_variance=False,
            )
        ),
        num_workers=min(maxthreads, 4),
        batch_size=8,
        pin_memory=False,
        shuffle=True,
        worker_init_fn=wif,
        prefetch_factor=8,
        drop_last=True,
        persistent_workers=True,
    )

    dlSelf = DataLoader(
        WasabiSelf(selfdata, selfmasks, 48),
        num_workers=min(maxthreads, 4),
        batch_size=6,
        pin_memory=False,
        shuffle=True,
        worker_init_fn=wif,
        prefetch_factor=8,
        drop_last=True,
        persistent_workers=True,
    )
    dPseudo = WasabiPseudoLabels(pseudolabels, pseudomask, fwdata)
    dlPseudo = DataLoader(
        dPseudo,
        num_workers=min(maxthreads, 4),
        batch_size=12,
        pin_memory=False,
        shuffle=True,
        worker_init_fn=wif,
        prefetch_factor=8,
        drop_last=True,
        persistent_workers=True,
    )

    dVal = Hdf5DataSet(validation_path)
    dlVal = DataLoader(
        dVal,
        num_workers=min(8, maxthreads),
        batch_size=32,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=wif,
        prefetch_factor=8,
    )
    fw = FWFakeR2(fwfunction(offset, trec + 0.0055, 42.5764 * 3))
    return (
        (dlSyn, dlPseudo, dlSelf),
        dlVal,
        fw,
        size,
        offset,
        trec,
    )
