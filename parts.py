from typing import Union, Optional, Tuple, Callable
from functools import partial
import torch
from torch import nn



class Concat(nn.Module):
    def __init__(self, crop=False):
        super().__init__()
        self.crop = crop

    def forward(self, x):
        if self.crop:
            minshape = tensor([np.inf] * (x[0].ndim - 2))
            for c in x:
                minshape = tuple(min(m, s) for m, s in zip(minshape, c.shape[2:]))
            xn = []
            for c in x:
                if c.shape[2:] > minshape:
                    newshape = (slice(None), slice(None)) + tuple((slice((s - m) // 2, (s - m) // 2 + m) for s, m in zip(c.shape[2:], minshape)))
                    xn.append(c[newshape])
                else:
                    xn.append(c)
            x = xn
        return torch.cat(x, 1)


def ConvNd(dim: int):
    return [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]


def ConvTransposeNd(dim: int):
    return [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dim - 1]


def NormNd(
    normtype: str,
    dim: int,
):
    if normtype == "batch":
        return [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dim - 1]
    if normtype == "instance":
        return [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d][dim - 1]
    if normtype == "layer":
        return nn.LayerNorm(*args, **kwargs)
    raise ValueError("normtype must be batch, instance or layer")


def DropoutNd(dim: int):
    return [nn.Dropout, nn.Dropout2d, nn.Dropout3d][dim - 1]


def MaxPoolNd(dim: int):
    return [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][dim - 1]


def AveragePoolNd(dim: int):
    return [nn.AveragePool1d, nn.AveragePool2d, nn.AveragePool3d][dim - 1]


class Residual(nn.Module):
    def __init__(self, block, residual=nn.Identity(), after = nn.Identity()):
        self.block = block
        self.residual = residual
        self.after = after
        
    def forward(self, x):
        return self.after(self.block(x) + self.residual(x))


class ResidualCBlock(nn.Module):
    def __init__(
        self,
        features: Tuple,
        dim: int = 2,
        kernel_size: Union[int, Tuple] = 3,
        dropout: Union[float, nn.Module, None] = None,
        norm: Union[bool, Callable[..., nn.Module], str] = False,
        norm_before_activation=True,
        activation: Optional[Callable[..., nn.Module]] = partial(nn.ReLU, inplace=True),
        bias: bool = True,
        padding: Union[bool, int] = True,
        padding_mode: str = "zeros",
        groups: int = 1,
        final_activation = True,
    ):
        """
        Convolutions from features[0]->features[1]->...->features[-1] with residual connection
        """
        self.block = CBlock(features, dim, kernelsize, dropout, norm, norm_before_activation, activation, bias, padding, padding_mode, groups, final_activation = None)
        self.resConv = ConvNd(dim)(kernel_size=1, bias=False)  if features[0] != features[-1] else None
        self.final_activaion = activation() if final_activation else None
    
    def forward(self, x):
        ret = self.block(x)
        if self.resConv:
            x = self.resConv(x)
        ret = ret + x
        if self.final_activation:
            ret = self.final_activation(ret)
        return ret
        
    
class CBlock(nn.Sequential):
    def __init__(
        self,
        features: Tuple,
        dim: int = 2,
        kernel_size: Union[int, Tuple] = 3,
        dropout: Union[float, nn.Module, None] = None,
        norm: Union[bool, Callable[..., nn.Module], str] = False,
        norm_before_activation=True,
        activation: Optional[Callable[..., nn.Module]] = partial(nn.ReLU, inplace=True),
        bias: bool = True,
        padding: Union[bool, int] = True,
        padding_mode: str = "zeros",
        groups: int = 1,
        final_activation = True,
        stride: int = 1
    ):
        """
        Convolutions from features[0]->features[1]->...->features[-1] with activation, optional norm and optional dropout 
        """
        

        if padding is True:
            if isinstance(kernel_size, (tuple, list)):
                padding = tuple((k // 2 for k in kernel_size))
            else:
                padding = kernel_size // 2
        if isinstance(dropout, float) and dropout > 0:
            dropout = DropoutNd(dim)(dropout)
        if isinstance(norm, str):
            norm = NormNd(normtype=norm, dim=dim)
        elif norm is True:
            norm = NormNd(normtype="batch", dim=dim)
        conv = partial(ConvNd(dim), kernel_size=kernel_size, padding=padding, groups=groups, bias=bias, padding_mode=padding_mode, stride=stride)
        modules = []
        for i, (fin, fout) in enumerate(zip(features[:-1], features[1:])):
            modules.append(conv(fin, fout))
            if dropout:
                modules.append(dropout)
            if norm and norm_before_activation:
                modules.append(norm(fout))
            if activation and (final_activation or i<len(features)-2):
                modules.append(activation())
            if norm and not norm_before_activation:
                modules.append(norm(fout))
        
        super().__init__(*modules)
        

    def __add__(self, other):
        new = type(self)(())
        for m in [*self, *other]:
            new.add_module(str(len(new)), m)
        return new
    
        

class Sequence(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.steps = nn.ModuleList(modules)

    def forward(self, x):
        ret = [x]
        for m in self.steps:
            x = m(x)
            ret.append(x)
        return ret

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp, in the backward pass like the identity function (gradient 1).
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/3
    """

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    input (Tensor): the input tensor.
    min (Number or Tensor, optional): lower-bound of the range to be clamped to
    max (Number or Tensor, optional): upper-bound of the range to be clamped to
    """
    return DifferentiableClamp.apply(input, min, max)