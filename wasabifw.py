# Forward Models for Wasabi
import abc
import numpy as np
import torch
from torch import nn, tensor


class WasabiMz(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, offset, trec, freq: float = 42.5764 * 3, tp: float = 0.005, gammabar: float = 42.5764, delay_after_pulse : float = 0.00553):
        """
        Wasabi Mz Forward Model
        """
        super().__init__()
        # allows .cuda() to move parameters. strictly not necessary for the scalar ones
        self.register_buffer("offset", torch.as_tensor(offset))
        self.register_buffer("trec", torch.as_tensor(trec))
        self.register_buffer("freq", tensor(freq * 2 * np.pi))
        self.register_buffer("gamma", tensor(gammabar * 2 * np.pi))
        self.register_buffer("tp", tensor(tp))
        self.register_buffer("delay_after_pulse", tensor(delay_after_pulse)) 

    @abc.abstractmethod
    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        pass


class WasabiMzExp(WasabiMz):
    """
    Base Class for MatrixExp based solutions
    """
    def prepare(self: WasabiMz, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the Matrix A, and the vectors c and mi
        """
        r1, r2, b0_shift, b1 = x
        device = self.offset.device
        zero = torch.tensor(0, device=device)
        delta = ((self.offset + b0_shift.unsqueeze(-1)) * self.freq).moveaxis(-1, 1)
        w1 = self.gamma * b1
        mi = torch.stack(
            torch.broadcast_tensors(zero, zero, -torch.expm1(-r1.unsqueeze(-1) * self.trec)),
            -1,
        ).moveaxis(-2, 1)
        A = torch.stack(
            torch.broadcast_tensors(
                -r2.unsqueeze(1), -delta, zero,
                delta, -r2.unsqueeze(1), w1.unsqueeze(1),
                zero, -w1.unsqueeze(1), -r1.unsqueeze(1),
            ),
            -1,
        ).reshape(r1.size()[:1] + self.trec.size() + r1.size()[1:] + torch.Size([3, 3]))

        c = torch.stack(torch.broadcast_tensors(zero, zero, r1.unsqueeze(1)), -1)
        return A, c, mi


class WasabiMzExpM(WasabiMzExp):
    def forward(self: WasabiMzExp, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Use Pytorch matrix_exp
        """
        A, c, mi = self.prepare(x)
        mexp = torch.matrix_exp(A * self.tp)
        ss = torch.linalg.solve(A, c.unsqueeze(-1))
        mz = torch.matmul(mexp[..., -1:, :], (mi.unsqueeze(-1) + ss))[..., 0, 0] - ss[..., -1, 0]
        mz += (mz - 1) * torch.expm1(-x[0] * self.delay_after_pulse).unsqueeze(1)
        return torch.abs(mz)


class WasabiMzExpEV(WasabiMzExp):
    def forward(self: WasabiMzExp, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Do an EV decompostion
        parameter x: R1, R2, b0_shift, b1
        """
        A, c, mi = self.prepare(x)
        L, V = torch.linalg.eig(A)
        ss = torch.linalg.solve(A, c.unsqueeze(-1))
        mexp = torch.real(
            torch.linalg.solve(
                V.swapaxes(-1, -2),
                (V * torch.exp(self.tp * L.unsqueeze(-2)))[..., -1, :],
            )
        )
        mz = torch.matmul(mexp.unsqueeze(-2), (mi.unsqueeze(-1) + ss))[..., 0, 0] - ss[..., -1, 0]
        mz += (mz - 1) * torch.expm1(-x[0] * self.delay_after_pulse).unsqueeze(1)
        return torch.abs(mz)


class WasabiMzApprox(WasabiMz):
    def forward(self: WasabiMz, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Use the closed form approximation
        parameter x: R1, R2, b0_shift, b1
        """
        r1, r2, b0_shift, b1 = x
        delta_2 = ((self.offset + b0_shift.unsqueeze(-1)).moveaxis(-1, 1) * self.freq) ** 2
        w1_2 = torch.clamp_((self.gamma * b1) ** 2, min=1e-10, max=1e10)
        deltaw1 = delta_2 + (w1_2).unsqueeze(1)
        r1p = (r1.unsqueeze(1) * delta_2 + (r2 * w1_2).unsqueeze(1)) / deltaw1
        r2p = r2.unsqueeze(1) + (r1.unsqueeze(1) - r1p) / 2
        minusmiz = torch.expm1((r1.unsqueeze(-1) * -self.trec).moveaxis(-1, 1))
        expr1p = torch.exp(r1p * -self.tp)

        minusmz = minusmiz * ((delta_2 * expr1p) + (w1_2.unsqueeze(1) * torch.cos(torch.sqrt(deltaw1) * self.tp)) * torch.exp(r2p * -self.tp)) / deltaw1 - (r1.unsqueeze(1) * delta_2) / (
            r1p * deltaw1
        ) * (1 - expr1p)
        minusmz += (minusmz + 1) * torch.expm1(-r1 * self.delay_after_pulse).unsqueeze(1)
        return torch.abs(minusmz)


class WasabiMzAna(WasabiMz):
    @staticmethod
    def _cbrt(x):
        return x.sign() * x.abs().pow(1 / 3)

    def forward(self: WasabiMz, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Use analytical solution
        parameter x: R1, R2, b0_shift, b1
        """
        r1, r2, b0_shift, b1 = x
        delta_2 = ((self.offset + b0_shift.unsqueeze(-1)).moveaxis(-1, 1) * self.freq) ** 2
        R1=r1.unsqueeze(1)
        R2=r2.unsqueeze(1)
        w1_2 = torch.clamp_((self.gamma * b1) ** 2, min=1e-10, max=1e10).unsqueeze(1)
        R2_2 = R2 ** 2
        R1R2 = R1 * R2
        Mzi = (1 - torch.exp(-r1.unsqueeze(-1) * self.trec)).moveaxis(-1, 1)
        
        p = (2 * R2 + R1)/3
        q = (R2_2 + 2 * R1R2 + w1_2) + delta_2
        r = (R1 * R2_2 + R2 * w1_2) + delta_2 * R1
        
        a = 1/3*q - p ** 2
        b = p ** 3 - 1/2 * (p*q-r)
        c = b ** 2 + a ** 3
        
        sqrtc = torch.sqrt(c)
        A = WasabiMzAna._cbrt((-b + sqrtc))
        B = WasabiMzAna._cbrt((-b - sqrtc))

        a1 = -p + A + B
        a2i = 1 / 2 * 3 ** (1 / 2) * (A - B)
        a2r = -p - (A + B) / 2
        a2 = torch.complex(a2r, a2i)
        m1 = (((R2 + a1) ** 2 + delta_2) * (R1 + Mzi * a1)) / (a1 * (a2i ** 2 + (a1 - a2r) ** 2))
        m2 = (((R2 + a2) ** 2 + delta_2) * (R1 + Mzi * a2)) / (a2 * (a2 - a1) * (torch.tensor(2j) * a2i))
        mss = (R1 * (delta_2 + R2_2)) / (R1 * (delta_2 + R2_2) + R2 * w1_2)

        mz = mss + m1 * torch.exp(a1 * self.tp) + 2 * torch.exp(a2r * self.tp) * (m2.real * torch.cos(a2i * self.tp) - m2.imag * torch.sin(a2i * self.tp))
        mz += (mz - 1) * torch.expm1(-R1 * self.delay_after_pulse)

        return torch.abs(mz)
