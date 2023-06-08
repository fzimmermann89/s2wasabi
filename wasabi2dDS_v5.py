from functools import partial, reduce
from itertools import chain
from pathlib import Path
import multiprocessing

import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from augments import correlatednoise, polynoise, randmult
from datasets import (
    BrainwebR1R2ClassesSlices,
    DataSetToHdf5,
    Hdf5DataSet,
    PhantomR1R2,
    WasabiDS,
)
from util import (
    AddFillMask,
    RemoveFillMask,
    cutnan,
    gamma_normal_noise,
    random_gaussians,
    random_poly2d,
    wif,
)
from wasabifw import WasabiMzApprox as fwfunction


def getLoader(path):

    maxthreads = multiprocessing.cpu_count()
    path = Path(path)
    size = (144, 144)
    offset = np.linspace(-2, 2, 31).astype(np.float32)
    trec = np.array([0.5, 1, 1.5, 2, 2.5, 3] + (len(offset) - 12) * [1.5] + [3, 2.5, 2, 1.5, 1, 0.5]).astype(np.float32)
    fwdata = torch.jit.script(fwfunction(offset, trec + 0.0055, 42.5764 * 3)).cpu()
    validation_path = path / 'validation5.hdf5'
    _field_blur = T.GaussianBlur(5, sigma=(0.5, 2))
    field_blur = lambda x: _field_blur(x[None,None,...])[0,0]
    transforms = (
        AddFillMask,
        T.GaussianBlur(7, sigma=(0.01, 2)),
        T.RandomAffine(30, scale=(0.3, 0.6), fill=0, shear=15, interpolation=T.InterpolationMode.BILINEAR),
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
                    [dict(strength=(0.25, 0.3), mode="add"), dict(strength=(0.06, 0.06), mode="mul"), dict(strength=(0.35, 0.075), mode="invadd")],
                )
            ],
            p=0.9,
        ),
        T.GaussianBlur(3, sigma=(0.01, 0.2)),
    )

    d = WasabiDS(
        torch.utils.data.ConcatDataset(
            (
                BrainwebR1R2ClassesSlices(path / 'train', axis=0, cuts=(96, 64, 16, 16, 16, 16), transforms=transforms),
                BrainwebR1R2ClassesSlices(path / 'train', axis=1, step=2, cuts=(32, 16, 128, 160, 16, 16), transforms=transforms),
                BrainwebR1R2ClassesSlices(path / 'train', axis=2, step=2, cuts=(32, 16, 16, 16, 128, 128), transforms=transforms),
                PhantomR1R2(size, 100),
            )
        ),
        (lambda *s: field_blur(correlatednoise(random_poly2d(*s, strength=(1.5, 0.009), p=0.99, sall=0.2), scale=[12, 32], strength=0.1))),
        (lambda *s: field_blur(correlatednoise(random_gaussians(*s, s=0.1, scales=(0.1, 0.3, 0.9), p=0.98, sall=0.1), scale=[16, 32], strength=0.02))),
        fwdata.cpu(),
        partial(gamma_normal_noise, meanvar=1.5e-2, varvar=6e-04, same_axes=((-1, -2, -3),)),
        return_variance=True,
    )
    try:
        dVal = Hdf5DataSet(validation_path)
    except FileNotFoundError:
        dVal = WasabiDS(
            torch.utils.data.ConcatDataset(
                (
                    BrainwebR1R2ClassesSlices(path / 'val', axis=0, cuts=(96, 64, 16, 16, 16, 16), transforms=transforms),
                    BrainwebR1R2ClassesSlices(path / 'val', axis=1, step=2, cuts=(32, 16, 128, 160, 16, 16), transforms=transforms),
                    BrainwebR1R2ClassesSlices(path / 'val', axis=2, step=2, cuts=(32, 16, 16, 16, 128, 128), transforms=transforms),
                )
            ),
            (lambda *s: field_blur(correlatednoise(random_poly2d(*s, strength=(1.5, 0.009), p=0.99, sall=0.2), scale=[12, 32], strength=0.1))),
            (lambda *s: field_blur(correlatednoise(random_gaussians(*s, s=0.1, scales=(0.1, 0.3, 0.9), p=0.95, sall=0.1), scale=[16, 32], strength=0.02))),
            fwdata.cpu(),
            partial(gamma_normal_noise, meanvar=1.5e-2, varvar=6e-04, same_axes=((-1, -2, -3),)),
            return_variance=True,
        )
        DataSetToHdf5(dVal, validation_path, verbose=True)
        dVal = Hdf5DataSet(validation_path)
    dl = DataLoader(
        d,
        num_workers=min(maxthreads, 16),
        batch_size=16,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=wif,
        prefetch_factor=8,
        drop_last=True,
        persistent_workers=True,
    )
    dlVal = DataLoader(dVal, num_workers=min(8, maxthreads), batch_size=32, pin_memory=True, shuffle=True, worker_init_fn=wif, prefetch_factor=8)
    fw = torch.jit.script(fwfunction(offset, trec + 0.0055, 42.5764 * 3))
    return (
        dl,
        dlVal,
        fw,
        size,
        offset,
        trec,
    )
