import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as snd
import torch
from torch import tensor
from torch.utils.data import Dataset
from torchvision import transforms as T
from util import cutnan, phantom, trunc_norm


@dataclass
class t1t2:
    t1min: float
    t1max: float
    t2min: float
    t2max: float


class BrainwebR1R2ClassesSlices(Dataset):
    def __init__(
        self,
        folder,
        cuts=(0, 0, 0, 0, 0, 0),
        axis=0,
        step=1,
        transforms=(
            T.RandomCrop((256, 256), pad_if_needed=True, fill=np.nan),
            T.RandomApply([partial(torch.rot90, dims=(-1, -2))]),
            T.RandomHorizontalFlip(),
            T.RandomHorizontalFlip(),
        ),
        classes=None,
    ):
        if classes is None:
            self.classes = {
                "gry": t1t2(1500, 2000, 80, 120),  #
                "wht": t1t2(900, 1500, 60, 100),
                "csf": t1t2(2800, 4500, 1300, 2000),
                "mrw": t1t2(400, 600, 60, 100),
                "dura": t1t2(2200, 2800, 200, 500),
                "fat": t1t2(300, 500, 60, 100),
                "fat2": t1t2(400, 600, 60, 100),
                "mus": t1t2(1200, 1500, 40, 60),
                "m-s": t1t2(500, 900, 300, 500),
                "ves": t1t2(1700, 2100, 200, 400),
            }
        else:
            self.classes = classes
        self._cuts = cuts
        self._axis = axis
        self._step = step
        files = []
        ns = [0]
        for fn in Path(folder).glob("*.h5"):
            try:
                with h5py.File(fn) as f:
                    ns.append((f["classes"].shape[self._axis]) - (self._cuts[self._axis * 2] + self._cuts[self._axis * 2 + 1]))
                    files.append(fn)
            except:
                pass
        self._files = tuple(files)
        self._ns = np.cumsum(ns)
        self._transforms = T.Compose(transforms)

    def __len__(self):
        return self._ns[-1] // self._step

    def __getitem__(self, index):
        if index * self._step >= self._ns[-1]:
            raise IndexError
        elif index < 0:
            index = self._ns[-1] + index * self._step
        else:
            index = index * self._step
        fileid = np.searchsorted(self._ns, index, "right") - 1
        sliceid = index - self._ns[fileid] + self._cuts[self._axis * 2]
        with h5py.File(self._files[fileid]) as f:
            where = [slice(self._cuts[2 * i], f["classes"].shape[i] - self._cuts[2 * i + 1]) for i in range(3)] + [slice(None)]
            where[self._axis] = sliceid
            data = np.array(f["classes"][tuple(where)], dtype=float)
            norm = np.array(f["norm"][tuple(where[:-1])], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r1v = np.array(
                    [1e3 / random.uniform(self.classes[i].t1min, self.classes[i].t1max) for i in f.attrs["classnames"] if i != "background"]
                )
                r2v = np.array(
                    [1e3 / random.uniform(self.classes[i].t2min, self.classes[i].t2max) for i in f.attrs["classnames"] if i != "background"]
                )
                cutslice = torch.from_numpy(cutnan(np.stack((data @ r1v, data @ r2v)) * norm[None, ...]))
        return self._transforms(cutslice)


class BrainwebR1R2Slices(Dataset):
    def __init__(
        self,
        folder,
        cuts=(0, 0, 0, 0, 0, 0),
        axis=0,
        transforms=(
            T.RandomCrop((256, 256), pad_if_needed=True, fill=np.nan),
            T.RandomApply([partial(torch.rot90, dims=(-1, -2))]),
            T.RandomHorizontalFlip(),
            T.RandomHorizontalFlip(),
        ),
    ):
        self._cuts = cuts
        self._axis = axis
        files = []
        ns = [0]
        for fn in Path(folder).glob("*.h5"):
            try:
                with h5py.File(fn, "r") as f:
                    if not f["r1"].shape == f["r2"].shape:
                        continue
                    ns.append((f["r1"].shape[axis]) - (self._cuts[self._axis * 2] + self._cuts[self._axis * 2 + 1]))
                    files.append(fn)
            except:
                pass
        self._files = tuple(files)
        self._ns = np.cumsum(ns)
        self._transforms = T.Compose(transforms)

    def __len__(self):
        return self._ns[-1]

    def __getitem__(self, index):
        if index >= self._ns[-1]:
            raise IndexError
        elif index < 0:
            index = self._ns[-1] + index
        fileid = np.searchsorted(self._ns, index, "right") - 1
        sliceid = index - self._ns[fileid] + self._cuts[self._axis * 2]
        with h5py.File(self._files[fileid], "r", rdcc_w0=1) as f:
            where = [slice(self._cuts[2 * i], f["r1"].shape[i] - self._cuts[2 * i + 1]) for i in range(3)]
            where[self._axis] = sliceid
            cutslice = torch.from_numpy(cutnan(np.stack([(np.array(f[what][tuple(where)])) for what in ("r1", "r2")])))
        return self._transforms(cutslice)


class WasabiDS(Dataset):
    def __init__(self, R1R2DataSet, b0_function, b1_function, fw, noise_function, maskdilation=0, return_variance=False):
        self._R1R2DataSet = R1R2DataSet
        self._b0_function = b0_function
        self._b1_function = b1_function
        self._noise_function = noise_function
        self.fw = fw.cpu()
        self.maskdilation = maskdilation
        self.return_variance = return_variance
        self._maskvaluesX = torch.tensor([0.8, 10, 0.0, 3.75])[:, None]
        self._maskvaluesY = self.fw(self._maskvaluesX).squeeze()[:, None]

    def __len__(self):
        return len(self._R1R2DataSet)

    def __getitem__(self, index):
        r1, r2 = torch.as_tensor(self._R1R2DataSet[index], dtype=torch.float32)
        b0 = torch.as_tensor(self._b0_function(*r1.shape), dtype=torch.float32)
        b1 = torch.as_tensor(3.75 * self._b1_function(*r1.shape), dtype=torch.float32)
        x = torch.stack((r1, r2, b0, b1))
        mask = torch.all(torch.isfinite(x), axis=0, keepdims=True)
        x[0] = torch.nan_to_num_(torch.clamp_(x[0], 1e-1, 20), nan=float(self._maskvaluesX[0]))
        x[1] = torch.nan_to_num_(torch.clamp_(x[1], 1e-1, 40), nan=float(self._maskvaluesX[1]))
        x[2] = torch.nan_to_num_(torch.clamp_(x[2], -2, 2), nan=float(self._maskvaluesX[2]))
        x[3] = torch.nan_to_num_(torch.clamp_(x[3], 0.25 * 3.75, 4 * 3.75), nan=float(self._maskvaluesX[3]))
        y = torch.nan_to_num(self.fw(x.unsqueeze(1)).squeeze(0).float())
        mask = torch.logical_and(torch.all(torch.isfinite(y), axis=0, keepdim=True), torch.as_tensor(mask))
        y = torch.clamp_(y, 0, 1.5)
        x[:, ~mask[0]] = self._maskvaluesX
        y[:, ~mask[0]] = self._maskvaluesY
        yn, sigma = self._noise_function(y, mask)
        yn = torch.clamp_(torch.abs_(yn), min=1e-10, max=5)
        if self.maskdilation > 0:
            mask = torch.as_tensor(~snd.binary_dilation(~mask, np.ones((1, self.maskdilation, self.maskdilation))))
        if self.return_variance:
            return x.float(), y.float(), yn.float(), mask.float(), ((sigma) ** 2).float()
        else:
            return x.float(), y.float(), yn.float(), mask.float()


class PhantomR1R2(Dataset):
    def __init__(self, size, epochsize=10, scale=(0.1, 0.4), maxn=10, radius=(0.7, 0.9)):
        self.epochsize = epochsize
        self.size = size
        self.scale = scale
        self.maxn = maxn
        self.radius = radius

    def __len__(self):
        return self.epochsize

    def __getitem__(self, index):
        return phantom(*self.size, self.scale, self.maxn, self.radius)


class bakedDS(Dataset):
    def __init__(self, DS):
        self.data = [sample for sample in tqdm(iter(DS))]

    def __len__(self):
        return len(self.daa)

    def __getitem__(self, index):
        return self.data[index]


def DataSetToHdf5(DS, path, verbose=False):
    from tqdm import tqdm

    with h5py.File(path, "x") as file:
        for ii, item in tqdm(enumerate(iter(DS))) if verbose else enumerate(iter(DS)):
            if isinstance(item, (list, tuple)):
                for ie, element in enumerate(item):
                    file[f"{ii}/{ie}"] = np.asarray(element)
                file[f"{ii}"].attrs["len"] = ie + 1
            elif isinstance(item, (np.ndarray, torch.Tensor)):
                file[f"{ii}"] = np.asarray(item)
                file[f"{ii}"].attrs["len"] = -1
        file.attrs["len"] = ii + 1


class Hdf5DataSet(Dataset):
    def __init__(self, path):
        super().__init__()
        self._len = h5py.File(path, "r").attrs["len"]
        self._path = path

    def __getitem__(self, index):
        if index >= self._len:
            raise IndexError
        if index < 0:
            index = self._len + index - 1
        with h5py.File(self._path, "r") as file:
            ds = file[f"{index}"]
            n = ds.attrs["len"]
            if n == -1:
                ret = np.asarray(ds)
            else:
                ret = tuple((np.asarray(ds[str(i)]) for i in range(n)))
        return ret

    def __len__(self):
        return self._len
