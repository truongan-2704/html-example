# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Adaptive Fractional Convolution (AFConv) modules for Ultralytics YOLOv11."""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.head import Detect


class AFConv2d(nn.Module):
    """Adaptive Fractional Convolution (AFConv2d) module simulating Grünwald-Letnikov fractional calculus.

    Attributes:
        alpha (nn.Parameter): Learnable fractional order parameter of shape [c2, 1, 1, 1], initialized to 1.0.
    """

    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, g: int = 1, d: int = 1, act=True):
        super().__init__()
        if isinstance(k, int):
            k_h = k_w = k
        else:
            k_h, k_w = k

        self.c1 = c1
        self.c2 = c2
        self.k_h = k_h
        self.k_w = k_w
        self.stride = s
        self.padding = autopad(k, p, d)
        self.dilation = d
        self.groups = g

        self.weight = nn.Parameter(torch.empty(c2, c1 // g, k_h, k_w))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Learnable alpha parameter [out_channels, 1, 1, 1] initialized to 1.0
        self.alpha = nn.Parameter(torch.ones(c2, 1, 1, 1))

        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # Precompute spatial Manhattan distance grid from kernel center
        c_h, c_w = k_h // 2, k_w // 2
        dist = []
        for r in range(k_h):
            row = []
            for c in range(k_w):
                row.append(abs(r - c_h) + abs(c - c_w))
            dist.append(row)
        self.dist = dist
        self.max_d = max(max(r) for r in dist)

    def compute_gl_kernel(self) -> torch.Tensor:
        """Computes Grünwald-Letnikov fractional coefficient weights w_k(alpha)."""
        w = [torch.ones_like(self.alpha)]
        for k in range(1, self.max_d + 1):
            w_k = w[-1] * (1.0 - (self.alpha + 1.0) / float(k))
            w.append(w_k)

        rows = []
        for r in range(self.k_h):
            cols = []
            for c in range(self.k_w):
                d_val = self.dist[r][c]
                cols.append(w[d_val])  # shape [c2, 1, 1, 1]
            rows.append(torch.cat(cols, dim=3))  # shape [c2, 1, 1, k_w]
        gl_kernel = torch.cat(rows, dim=2)  # shape [c2, 1, k_h, k_w]
        return gl_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gl_kernel = self.compute_gl_kernel()
        weight_eff = self.weight * gl_kernel
        out = F.conv2d(
            x,
            weight_eff,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return self.act(self.bn(out))


class ChannelAttention(nn.Module):
    """Channel-attention module for CBAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r_channels, channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.act(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial-attention module for CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        return x * self.act(self.cv1(scale))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, c1: int, c2: int = None, kernel_size: int = 7, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))


class Bottleneck_AF(nn.Module):
    """Standard bottleneck replacing 3x3 convolutions with AFConv2d."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = AFConv2d(c1, c_, k[0], 1) if k[0] == 3 else Conv(c1, c_, k[0], 1)
        self.cv2 = AFConv2d(c_, c2, k[1], 1, g=g) if k[1] == 3 else Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_AF(nn.Module):
    """Faster Implementation of CSP Bottleneck using AFConv2d (fractional convolutions)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_AF(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AF_Detect(Detect):
    """YOLO AF_Detect head overriding box regression branch (cv2) with AFConv2d."""

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        # Override self.cv2 to use AFConv2d for bounding box regression
        self.cv2 = nn.ModuleList(
            nn.Sequential(AFConv2d(x, c2, 3), AFConv2d(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
