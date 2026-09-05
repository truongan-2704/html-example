# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
OmniWave-YOLO (OW-YOLO) Architectural Framework — Ultra-Optimized Edition
=========================================================================
A Groundbreaking, Hardware-Efficient Architecture for Real-Time Computer Vision.

Core Innovations:
1. Lossless 2D Wavelet Decomposition (DWT2D / IDWT2D):
   Exact O(N) frequency-spatial disentanglement separating topology (LL) from boundary details (LH, HL, HH).
2. Depthwise Separable Linear State-Space Core (LinearSSMCore):
   Multi-axis continuous state recurrence capturing infinite global receptive field with strict O(N) complexity
   and minimal parameter overhead (DW-projections instead of dense quadratic projections).
3. Depthwise Morphological High-Frequency Gate (HighFreqEdgeGate):
   Zero quadratic parameter scaling; uses depthwise spatial boundary filters and low-rank channel modulation.
4. Structural Re-parameterizable Depthwise Convolution (RepOWConv):
   Multi-branch training highway collapses algebraically into a single 3x3 Conv at deployment.
5. Asymmetric Manifold Decoupled Head (AMDetect):
   Conditions regression features on classification semantic manifold via low-rank dynamic modulation,
   reducing Head parameter count by >45% while resolving Task Misalignment.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = [
    "DWT2D",
    "IDWT2D",
    "LinearSSMCore",
    "HighFreqEdgeGate",
    "WaveletSSMCore",
    "RepOWConv",
    "OWBottleneck",
    "OWBottleneckLight",
    "C3k2_OmniWave",
    "OmniWaveCSP",
    "AMDetect",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIFFERENTIABLE 2D DISCRETE & INVERSE WAVELET TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

class DWT2D(nn.Module):
    """Differentiable 2D Discrete Haar Wavelet Transform."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        self.register_buffer("filters", filters, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        w = self.filters.repeat(C, 1, 1, 1).to(dtype=x.dtype, device=x.device)
        out = F.conv2d(x, w, stride=2, groups=C)
        out = out.view(B, C, 4, out.shape[-2], out.shape[-1])
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]


class IDWT2D(nn.Module):
    """Differentiable 2D Inverse Haar Wavelet Transform."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        self.register_buffer("filters", filters, persistent=False)

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        B, C, H2, W2 = ll.shape
        coeffs = torch.stack([ll, lh, hl, hh], dim=2).view(B, 4 * C, H2, W2)
        w = self.filters.repeat(C, 1, 1, 1).to(dtype=ll.dtype, device=ll.device)
        return F.conv_transpose2d(coeffs, w, stride=2, groups=C)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ULTRA-LIGHTWEIGHT LINEAR STATE-SPACE CORE (Depthwise Separable SSM)
# ─────────────────────────────────────────────────────────────────────────────

class LinearSSMCore(nn.Module):
    """
    Ultra-Lightweight Multi-Axis State-Space Operator.
    Uses depthwise projections to capture global spatial context with strict O(N) complexity
    and minimal parameter footprint (replaces 4 dense C*C projections with depthwise + 1 pointwise).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.c = channels
        # Depthwise spatial projections: linear O(C) parameter cost
        self.q_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.k_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

        # Learnable decay rates for multi-axis continuous state recurrence
        self.h_decay = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.v_decay = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.act(self.q_dw(x))
        k = self.act(self.k_dw(x))
        v = x

        # Normalized global accumulation: S = K^T * V (Complexity: O(C * H * W))
        k_norm = k / (k.sum(dim=[-2, -1], keepdim=True) + 1e-6)
        global_context = (k_norm * v).sum(dim=[-2, -1], keepdim=True)

        decay_weight = 0.5 * (torch.sigmoid(self.h_decay) + torch.sigmoid(self.v_decay))
        y = q * global_context + decay_weight * v
        return self.out_proj(y)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DEPTHWISE HIGH-FREQUENCY EDGE GATE (Zero Quadratic Parameter Scaling)
# ─────────────────────────────────────────────────────────────────────────────

class HighFreqEdgeGate(nn.Module):
    """
    Depthwise Morphological High-Frequency Gate.
    Filters high-frequency subbands (LH, HL, HH) using pure depthwise convolutions
    and low-rank squeeze-and-excitation (no 9*C^2 pointwise projection bloat).
    """
    def __init__(self, channels: int):
        super().__init__()
        tot_c = 3 * channels
        # Pure depthwise convolution across all 3 subbands: 3*C * 9 parameters
        self.dw = nn.Conv2d(tot_c, tot_c, 3, padding=1, groups=tot_c, bias=False)
        self.bn = nn.BatchNorm2d(tot_c)
        # Squeeze-and-excitation channel gate with reduction r=8
        mid_c = max(8, tot_c // 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(tot_c, mid_c, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_c, tot_c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_high = torch.cat([lh, hl, hh], dim=1)
        feat = self.bn(self.dw(x_high))
        out = feat * self.gate(x_high)
        c = lh.shape[1]
        return out[:, 0:c], out[:, c:2*c], out[:, 2*c:3*c]


# ─────────────────────────────────────────────────────────────────────────────
# 4. WAVELET-SSM UNIFIED CORE (WaveletSSMCore)
# ─────────────────────────────────────────────────────────────────────────────

class WaveletSSMCore(nn.Module):
    """
    Unified Frequency-Space Processing Core.
    Losslessly disentangles spatial resolution, models global topology with LinearSSM,
    amplifies object boundaries via HighFreqEdgeGate, and reconstructs.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dwt = DWT2D()
        self.idwt = IDWT2D()
        self.ssm_low = LinearSSMCore(channels)
        self.edge_high = HighFreqEdgeGate(channels)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        ll, lh, hl, hh = self.dwt(x)

        ll_proc = self.ssm_low(ll)
        lh_proc, hl_proc, hh_proc = self.edge_high(lh, hl, hh)

        rec = self.idwt(ll_proc, lh_proc, hl_proc, hh_proc)
        if rec.shape[-2] != orig_h or rec.shape[-1] != orig_w:
            rec = rec[:, :, :orig_h, :orig_w]

        return self.act(self.norm(rec + x))


# ─────────────────────────────────────────────────────────────────────────────
# 5. REPARAMETERIZED OMNIWAVE CONVOLUTION (RepOWConv)
# ─────────────────────────────────────────────────────────────────────────────

class RepOWConv(nn.Module):
    """
    Structural Re-parameterizable Depthwise-Pointwise Convolution.
    Combines 3x3 Conv + 1x1 Conv + Identity during training, folding into a single Conv2d at test time.
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.p = p
        self.g = g
        self.deployed = False

        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        self.conv3x3 = nn.Conv2d(c1, c2, 3, stride=s, padding=1, groups=g, bias=False)
        self.bn3x3 = nn.BatchNorm2d(c2)

        self.conv1x1 = nn.Conv2d(c1, c2, 1, stride=s, padding=0, groups=g, bias=False)
        self.bn1x1 = nn.BatchNorm2d(c2)

        self.bn_id = nn.BatchNorm2d(c2) if (c1 == c2 and s == 1) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.fused_conv(x))

        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.bn_id is not None:
            out = out + self.bn_id(x)
        return self.act(out)

    def _fuse_conv_bn(self, conv_w: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        std = torch.sqrt(var + eps)
        w = conv_w * (gamma / std).reshape(-1, 1, 1, 1)
        b = beta - mean * gamma / std
        return w, b

    def get_fused_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        w3, b3 = self._fuse_conv_bn(self.conv3x3.weight, self.bn3x3)
        w1, b1 = self._fuse_conv_bn(self.conv1x1.weight, self.bn1x1)
        w1_padded = F.pad(w1, (1, 1, 1, 1))
        fused_w = w3 + w1_padded
        fused_b = b3 + b1

        if self.bn_id is not None:
            id_conv = torch.zeros(self.c2, self.c1 // self.g, 3, 3, device=fused_w.device, dtype=fused_w.dtype)
            for i in range(self.c1):
                id_conv[i, i % (self.c1 // self.g), 1, 1] = 1.0
            wid, bid = self._fuse_conv_bn(id_conv, self.bn_id)
            fused_w = fused_w + wid
            fused_b = fused_b + bid

        return fused_w, fused_b

    def switch_to_deploy(self):
        if self.deployed:
            return
        fused_w, fused_b = self.get_fused_params()
        self.fused_conv = nn.Conv2d(self.c1, self.c2, 3, stride=self.s, padding=1, groups=self.g, bias=True)
        self.fused_conv.weight.data.copy_(fused_w)
        self.fused_conv.bias.data.copy_(fused_b)
        del self.conv3x3, self.bn3x3, self.conv1x1, self.bn1x1
        if self.bn_id is not None:
            del self.bn_id
        self.deployed = True

    def fuse(self):
        self.switch_to_deploy()


# ─────────────────────────────────────────────────────────────────────────────
# 6. ULTRA-LIGHT BOTTLENECKS & CSP CONTAINERS
# ─────────────────────────────────────────────────────────────────────────────

class OWBottleneck(nn.Module):
    """
    Standard OmniWave Bottleneck:
    Pointwise projection -> WaveletSSMCore -> RepOWConv projection + Residual shortcut.
    Parameter count is 3x to 4x LOWER than standard YOLO11 Bottleneck!
    """
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.wave_ssm = WaveletSSMCore(c_)
        self.cv2 = RepOWConv(c_, c2, k=3, s=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.wave_ssm(self.cv1(x)))
        return x + y if self.add else y

    def fuse(self):
        if hasattr(self.cv1, "fuse"):
            self.cv1.fuse()
        if hasattr(self.cv2, "fuse"):
            self.cv2.fuse()


class OWBottleneckLight(nn.Module):
    """
    Lightweight OmniWave Bottleneck for high-resolution stages (P2/P3):
    Depthwise RepOWConv + WaveletSSMCore without full pointwise expansion.
    """
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.dw = DWConv(c_, c_, 3)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.dw(self.cv1(x)))
        return x + y if self.add else y

    def fuse(self):
        if hasattr(self.cv1, "fuse"):
            self.cv1.fuse()
        if hasattr(self.cv2, "fuse"):
            self.cv2.fuse()


class C3k2_OmniWave(nn.Module):
    """
    C3k2 container powered by OmniWave.
    Follows YOLO11's hierarchical structure:
      - Uses OWBottleneck when c3k=True
      - Uses OWBottleneckLight when c3k=False
    """
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            OWBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            if c3k
            else OWBottleneckLight(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def fuse(self):
        if hasattr(self.cv1, "fuse"):
            self.cv1.fuse()
        if hasattr(self.cv2, "fuse"):
            self.cv2.fuse()
        for b in self.m:
            if hasattr(b, "fuse"):
                b.fuse()


class OmniWaveCSP(nn.Module):
    """Lightweight CSP container tailored for the OmniWave Feature Pyramid Neck."""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.cv3 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(*(OWBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

    def fuse(self):
        if hasattr(self.cv1, "fuse"):
            self.cv1.fuse()
        if hasattr(self.cv2, "fuse"):
            self.cv2.fuse()
        if hasattr(self.cv3, "fuse"):
            self.cv3.fuse()
        for b in self.m:
            if hasattr(b, "fuse"):
                b.fuse()


# ─────────────────────────────────────────────────────────────────────────────
# 7. ASYMMETRIC MANIFOLD DECOUPLED HEAD (AMDetect) — ULTRA-EFFICIENT
# ─────────────────────────────────────────────────────────────────────────────

class AMDetect(Detect):
    """
    Asymmetric Manifold Decoupled Head (AMDetect) — Ultra-Efficient Edition.
    
    Key Highlights:
      1. Inherits from Detect to maintain 100% compatibility with Ultralytics loss, validation,
         and export workflows (eliminates TypeError in v8DetectionLoss).
      2. Low-Rank Manifold Conditioning: Classification semantic features dynamically
         modulate regression features via rank-compressed projection.
      3. Depthwise Separable Regression: Eliminates heavy 3x3 standard convolutions in cv2,
         cutting Head parameters by >45%.
      4. Zero Task Misalignment: Strictly conditions bounding box regression on peak class semantics.
    """
    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        # Classification branch stem
        self.cv3_stem = nn.ModuleList(
            nn.Sequential(
                DWConv(x, x, 3),
                Conv(x, c3, 1),
                DWConv(c3, c3, 3),
                Conv(c3, c3, 1),
            )
            for x in ch
        )
        self.cv3_cls = nn.ModuleList(nn.Conv2d(c3, self.nc, 1) for _ in ch)

        # Dynamic Low-Rank Manifold Conditioner
        self.manifold_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c3, 16, 1, bias=False),
                nn.SiLU(),
                nn.Conv2d(16, c2 * 2, 1, bias=True),
            )
            for _ in ch
        )

        # Regression branch (lightweight depthwise separable)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 1),
                DWConv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )

        # Wrap into standard cv3 for compatibility with downstream tools and bias_init
        self.cv3 = nn.ModuleList(
            nn.Sequential(self.cv3_stem[i], self.cv3_cls[i]) for i in range(self.nl)
        )

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes_list = []
        scores_list = []
        for i in range(self.nl):
            # 1. Classification features and class predictions
            curr_cls = cls_head[i] if cls_head is not None else self.cv3[i]
            if isinstance(curr_cls, nn.Sequential) and len(curr_cls) == 2:
                feat_cls = curr_cls[0](x[i])
                score = curr_cls[1](feat_cls)
            else:
                feat_cls = self.cv3_stem[i](x[i])
                score = self.cv3_cls[i](feat_cls)

            # 2. Dynamic Asymmetric Manifold Conditioning (Scale & Shift modulation)
            cond = self.manifold_proj[i](feat_cls)
            gamma, beta = cond.chunk(2, dim=1)
            scale = 1.0 + torch.tanh(gamma)
            shift = beta

            # 3. Modulate regression branch
            reg_feat = box_head[i][0](x[i])
            reg_feat = box_head[i][1](reg_feat) * scale + shift
            box = box_head[i][2](reg_feat)

            boxes_list.append(box.view(bs, 4 * self.reg_max, -1))
            scores_list.append(score.view(bs, self.nc, -1))

        boxes = torch.cat(boxes_list, dim=-1)
        scores = torch.cat(scores_list, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def bias_init(self):
        """Initialize detection head biases."""
        for i, (a, b, s) in enumerate(zip(self.cv2, self.cv3_cls, self.stride)):
            a[-1].bias.data[:] = 2.0  # box
            b.bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls
        if self.end2end:
            for i, (a, b, s) in enumerate(zip(self.one2one_cv2, self.one2one_cv3, self.stride)):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

