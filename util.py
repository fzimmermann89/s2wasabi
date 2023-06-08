import numpy as np
from typing import Tuple, Union, List, Optional
import torch
from torch import tensor, Tensor
from pathlib import Path
import math
import multiprocessing as mp



def trunc_norm(
    mu: Union[float, Tensor], sigma: Union[float, Tensor], a: Union[float, Tensor], b: Union[float, Tensor], size: Tuple[int] = (1,)
) -> Tensor:
    normal = torch.distributions.normal.Normal(0, 1)
    alpha = torch.as_tensor((a - mu) / sigma)
    beta = torch.as_tensor((b - mu) / sigma)
    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * torch.rand(size)
    v = torch.clip(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
    x = torch.clamp(x, a, b)
    return x


def wif(id: int):
    """
    Seed numpy in workers
    """
    np.random.seed((id + torch.initial_seed()) % np.iinfo(np.int32).max)


def cutnan(array: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    remove full-nan rows and columns (last two dimensions) of array
    """
    ind0 = ~np.all(np.isnan(np.asarray(array)), axis=tuple(s for s in range(array.ndim - 1)))
    ind1 = ~np.all(np.isnan(np.asarray(array)), axis=tuple(s for s in range(array.ndim - 2)) + (array.ndim - 1,))
    return array[..., ind1, :][..., ind0]


def random_poly2d(Nx: int, Ny: int, strength: Tuple[float], p: float = 1, sall: float = 0) -> np.ndarray:
    """
    Random 2d Polynom
    """
    if np.random.rand() > p:
        return np.zeros((Nx, Ny))
    order = np.size(strength)
    c = np.random.uniform(-1 / 2, 1 / 2, (2, order)) * np.asarray(strength)
    x = np.linspace(-1, 1, Nx)[:, None]
    y = np.linspace(-1, 1, Ny)[None, :]
    ret = (sum(c[0, i] * x ** (i + 1) for i in range(order)) + 1) * (sum(c[1, i] * y ** (i + 1) for i in range(order)) + 1) - 1
    if sall > 0:
        ret += sall * float(trunc_norm(0, sall, -2 * sall, 2 * sall))
    return ret


def random_gaussians(Nx: int, Ny: int, s: float, scales: Tuple[float], p: float = 1, sall: float = 0.0) -> np.ndarray:
    """
    Random Gaussians
    """
    allg = np.zeros((Nx, Ny))
    if np.random.rand() > p:
        return allg + 1
    for scalehigh, scalelow in zip(scales[1:], scales[:-1]):
        strength = np.random.uniform(-1, 1)
        sx, sy = np.random.uniform(scalelow, scalehigh, 2)
        mx, my = np.random.uniform(-1 + scalelow, 1 - scalelow, 2)
        alpha = np.random.uniform(0, np.pi / 2)
        x = np.linspace(-1, 1, Nx).reshape(-1, 1) - mx
        y = np.linspace(-1, 1, Ny).reshape(1, -1) - my
        g = np.exp(-((np.cos(alpha) * x - np.sin(alpha) * y) ** 2 / sx + (np.sin(alpha) * x + np.cos(alpha) * y) ** 2 / sy))

        allg += g * strength * (sx * sy)
    rand = -np.inf
    while np.abs(rand) > 2:
        rand = np.random.normal(size=1)
    allg *= rand * s / np.std(allg)
    allg -= np.mean(allg) - 1
    allg = np.clip(allg, 1 - 3 * s, 1 + 3 * s)
    if sall > 0:
        allg *= float(trunc_norm(1, sall, 1 - 2 * sall, 1 + 2 * sall, 1))
    allg = np.clip(allg, max(1e-2, 1 - 3 * s - 2 * sall), min(100, 1 + 3 * s + 2 * sall))
    return allg


def phantom(Nx: int, Ny: int, scale: Tuple[float] = (0.1, 0.4), maxn: int = 2, radius: Tuple[float] = (0.7, 1.0), nan_border: bool = True) -> Tensor:
    """
    :param Nx: first dimension
    :param Ny: second dimension
    :param scale: relative size of ellipses, (min, max)
    :param maxn: maximum number of ellipses
    :param radius: relative outer radius, (min, max)
    :param nan_border: border of nan values around ellipses
    :return: Tensor of r1,r2
    """

    x, y = torch.meshgrid(torch.linspace(-Nx // 2, Nx // 2 - 1, Nx), torch.linspace(-Ny // 2, Ny // 2 - 1, Ny))
    out = torch.zeros((2, Nx, Ny))
    out[0, :] = 1 / trunc_norm(2, 1, 0.1, 4)
    out[1, :] = 1 / trunc_norm(0.1, 0.1, 0.02, 1)
    cs = [(np.inf, np.inf)]
    rs = [(0)]
    for i in range(maxn):
        theta = torch.rand(1) * 2 * np.pi
        a = (torch.rand(1) * (scale[1] - scale[0]) + scale[0]) * min(Nx, Ny) / 2
        b = torch.clamp((1 + torch.randn(1) * 0.5), 0.5, 2) * a
        for retry in range(10 * maxn):
            c = (torch.rand(size=(2,)) - 0.5) * (tensor((Nx, Ny)) * (1 - scale[1]))
            dist = torch.sqrt(((tensor(cs) - c) ** 2).sum(1))
            if torch.min(dist - tensor(rs)) > (max(abs(a), abs(b))):
                break
        else:
            continue
        cs.append(c)
        rs.append(max(abs(a), abs(b)))
        cx, cy = c
        ellipse = ((x - cx) * torch.cos(theta) + (y - cy) * torch.sin(theta)) ** 2 / a ** 2 + (
            (x - cx) * torch.sin(theta) - (y - cy) * torch.cos(theta)
        ) ** 2 / b ** 2
        if nan_border:
            thickness = trunc_norm(15, 15, 10, 30) / min(Nx, Ny)
            out[:, (ellipse < 1 + thickness) & (ellipse > 1)] = np.nan
        out[0, ellipse < 1] = 1 / trunc_norm(2, 1, 0.1, 4)
        out[1, ellipse < 1] = 1 / trunc_norm(0.1, 0.1, 0.02, 1)
    r = torch.empty(1).uniform_(*radius)
    out[:, x ** 2 + y ** 2 > (r * max(Nx, Ny) / 2) ** 2] = np.nan
    return out


def wif(id: int):
    """
    Seed numpy in workers
    """
    np.random.seed((id + torch.initial_seed() + torch.randint(2 ** 31, (1,)).item()) % np.iinfo(np.int32).max)


def load_wasabidata(filename: Union[str, Path], thres: Optional[float] = None, fill_holes: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    import nibabel
    from skimage.filters import threshold_otsu
    import scipy.ndimage as snd

    img = nibabel.load(filename)
    data = np.array(img.get_fdata())
    if thres is None:
        thres = threshold_otsu(data[..., 0].ravel())
    mask = data[..., 0] > thres
    if fill_holes:
        mask = snd.binary_fill_holes(mask)
    data = data[..., 1:] / data[..., :1]
    data = np.transpose(data, axes=(-1, 0, 1))
    return data, mask


try:
    import bmctool.params, bmctool.bmc_tool
except ImportError:
    pass
else:
    class BMC:
        def __init__(
            self,
            seq_file: str = "/data/zimmer08/pycharm/Playground/WASABInet/library/20210706_WASABITI_sweep12_sim.seq",
            gamma: float = 42.58 * 2 * np.pi,
        ):
            params = bmctool.params.Params(False)
            params.set_options(max_pulse_samples=300, scale=False)
            params.set_water_pool(1, 1, 1)
            params.set_scanner(b0=0, gamma=gamma, b0_inhom=0, rel_b1=1)
            params.set_m_vec()
            self.Sim = bmctool.bmc_tool.BMCTool(params, seq_file, verbose=False)
            self.b1 = self.Sim.seq.get_definition("b1cwpe")
            self.b0 = self.Sim.seq.get_definition("b0")
            self.Sim.params.update_scanner(b0=self.b0)

        def sim(self, x: Union[np.ndarray, List, Tuple]) -> np.ndarray:
            ret = []
            for r1, r2, b0_inhom, b1 in np.atleast_2d(x):
                self.Sim.params.update_scanner(b0_inhom=b0_inhom, rel_b1=b1 / self.b1)
                self.Sim.params.update_water_pool(r1=r1, r2=r2)
                self.Sim.run()
                ret.append(self.Sim.get_zspec(return_abs=True)[1])
            return np.array(ret)


    def parallel_BMC(
        params: np.ndarray, seqfile: str = "/data/zimmer08/pycharm/Playground/WASABInet/library/20210706_WASABITI_sweep12_sim.seq", num_workers: int = 16
    ) -> np.ndarray:
        with mp.Pool(num_workers) as pool:
            tmp = pool.map(BMC(seqfile).sim, np.array_split(params, num_workers))
        return np.concatenate(tmp)


def gamma_normal_noise(image: Tensor, mask: Tensor, meanvar: float = 1e-2, varvar: float = 2e-5, same_axes: Tuple[Tuple[int]] = ((-1, -2), (-3))):
    m = torch.as_tensor(meanvar, dtype=torch.float64)
    v = torch.as_tensor(varvar, dtype=torch.float64) * len(same_axes)
    d = torch.distributions.gamma.Gamma(m ** 2 / v, m / v)
    var = []
    for sa in same_axes:
        s = tensor(image.shape)
        s[(sa,)] = 1
        var.append(d.sample(s))
    var = sum(var) / len(var)
    sigma = (mask > 0) * torch.sqrt(var)
    noise = (torch.randn(image.shape) * sigma).float()
    return noise + torch.nan_to_num(image, 0, 0, 0), sigma


def AddFillMask(x):
    mask = torch.all(torch.isfinite(x), 0, True)
    return torch.cat((torch.nan_to_num(x, 0, 0, 0), mask), 0)


def RemoveFillMask(x):
    mask = x[-1:, ...] < 0.95
    x = x[:-1, ...]
    x[mask.expand(x.shape)] = np.nan
    return x
