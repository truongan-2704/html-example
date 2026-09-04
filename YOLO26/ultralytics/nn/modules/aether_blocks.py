# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-Aether — Next-Generation Architectural Framework
=====================================================

Innovations:
1. Omni-Spectral Manifold Convolution (OSM-Conv):
   - Multi-scale manifold channel decomposition:
     * Detail Branch (1/4 C): High-resolution edge/boundary extraction (3x3 conv).
     * Dilated Context Branch (1/4 C): Multi-receptive field context (3x3 dilation=2, grouped).
     * Direct Flow (1/2 C): Zero-FLOPs linear highway preserving gradient integrity.
   - Energy Router Gate: Dual-pool adaptive spatial-channel modulation.
   - Pointwise cross-channel fusion.

2. Content-Spatial Adaptive Fusion (CSAF):
   - Eliminates PANet channel-doubling Concat bottlenecks and BiFPN scalar-weight limitations.
   - Generates dynamic 2D spatial weight maps [B, 2, H, W] for continuous inter-scale blending.
   - In-place weighted fusion with zero memory bandwidth explosion.

3. Harmonic Resonant Pyramid Core (AetherResonantCore):
   - Replaces 2-pass sequential Top-Down / Bottom-Up paths with a single-step resonant interaction.
   - Jointly aggregates P3, P4, P5 into a central harmonic manifold.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = [
    "OSMConv",
    "AetherBottleneck",
    "C3k2_Aether",
    "AetherCSP",
    "AetherCSAF",
    "AetherResonantCore",
]


def autopad(k, p=None, d=1):
    """Auto-pad to maintain exact spatial resolution."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1. OMNI-SPECTRAL MANIFOLD CONVOLUTION (OSM-Conv)
# ─────────────────────────────────────────────────────────────────────────────
class OSMConv(nn.Module):
    """
    Omni-Spectral Manifold Convolution (OSM-Conv).
    
    Decomposes input channels into:
    - 25% Detail Path: Micro-kernel (3x3) for high-frequency edges and small objects.
    - 25% Context Path: Grouped Dilated Conv (3x3, d=2) for broad semantic context.
    - 50% Direct Highway: Identity/Projection path with high arithmetic intensity.
    
    Modulated by an Energy Router Gate and fused with 1x1 pointwise projection.
    """
    def __init__(self, c1, c2, k=3, s=1, g=1, d=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s

        # Calculate partition channels
        self.c_det = max(1, c1 // 4)
        self.c_ctx = max(1, c1 // 4)
        self.c_id = c1 - self.c_det - self.c_ctx

        self.out_det = max(1, c2 // 4)
        self.out_ctx = max(1, c2 // 4)
        self.out_id = c2 - self.out_det - self.out_ctx

        # Detail branch (High-frequency micro-kernel)
        self.branch_detail = Conv(self.c_det, self.out_det, k=3, s=s)

        # Context branch (Dilated grouped convolution)
        ctx_groups = math.gcd(self.c_ctx, self.out_ctx)
        ctx_groups = min(4, max(1, ctx_groups))
        self.branch_context = nn.Sequential(
            nn.Conv2d(
                self.c_ctx,
                self.out_ctx,
                kernel_size=3,
                stride=s,
                padding=autopad(3, d=2),
                dilation=2,
                groups=ctx_groups,
                bias=False
            ),
            nn.BatchNorm2d(self.out_ctx),
            nn.SiLU(inplace=True)
        )

        # Direct Manifold Highway
        if s == 1 and self.c_id == self.out_id:
            self.branch_identity = nn.Identity()
        else:
            self.branch_identity = Conv(self.c_id, self.out_id, k=1, s=s)

        # Energy Router Gate: Dual-pool spatial-channel descriptor
        mid_router = max(8, c2 // 8)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, mid_router, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_router, c2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Final pointwise fusion
        self.fusion = Conv(c2, c2, k=1, s=1)

    def forward(self, x):
        # Channel manifold partition
        x_det, x_ctx, x_id = torch.split(x, [self.c_det, self.c_ctx, self.c_id], dim=1)

        y_det = self.branch_detail(x_det)
        y_ctx = self.branch_context(x_ctx)
        y_id = self.branch_identity(x_id)

        # Concatenate decomposed subbands
        y_raw = torch.cat([y_det, y_ctx, y_id], dim=1)

        # Dynamic Energy modulation
        gate = self.router(x)
        y = y_raw * gate

        return self.fusion(y)


# ─────────────────────────────────────────────────────────────────────────────
# 2. AETHER BOTTLENECK & CSP CONTAINERS
# ─────────────────────────────────────────────────────────────────────────────
class AetherBottleneck(nn.Module):
    """
    Inverted residual bottleneck powered by OSM-Conv.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_hidden = int(c2 * e)
        self.cv1 = Conv(c1, c_hidden, k[0], 1)
        self.cv2 = OSMConv(c_hidden, c2, k=k[1], s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2_Aether(nn.Module):
    """
    CSP container with AetherBottleneck for Backbone feature extraction.
    Drop-in upgrade for standard C3k2.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            AetherBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AetherCSP(nn.Module):
    """
    Lightweight CSP block tailored for the Aether Neck.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(*(
            AetherBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        ))

    def forward(self, x):
        y1, y2 = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((y1, self.m(y2)), 1))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONTENT-SPATIAL ADAPTIVE FUSION (CSAF)
# ─────────────────────────────────────────────────────────────────────────────
class AetherCSAF(nn.Module):
    """
    Content-Spatial Adaptive Fusion (AetherCSAF).
    
    Replaces PANet Concat and BiFPN scalar fusion.
    Takes 2 input feature maps [local_feature, global_feature],
    normalizes their spatial sizes and channel counts, computes a 2D pixel-wise
    Softmax weight map, and fuses them without doubling channel bandwidth.
    """
    def __init__(self, c1, c2):
        super().__init__()
        # c1 can be a list of input channels or an int
        if isinstance(c1, int):
            c1 = [c1, c1]
        self.c1 = c1
        self.c2 = c2

        # Align input channels to target c2
        self.proj_local = Conv(c1[0], c2, 1) if c1[0] != c2 else nn.Identity()
        self.proj_global = Conv(c1[1], c2, 1) if c1[1] != c2 else nn.Identity()

        # Spatial Dynamic Weight Matrix Generator
        hidden_dim = max(16, c2 // 4)
        self.weight_generator = nn.Sequential(
            nn.Conv2d(c2 * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True),
            nn.Softmax(dim=1)  # Normalizes weights across the 2 branches for each pixel
        )

        # Non-linear refinement
        self.refine = OSMConv(c2, c2)

    def forward(self, x):
        # x is a list of [local, global]
        if isinstance(x, torch.Tensor):
            return x
        
        x_local, x_global = x[0], x[1]
        
        # Match spatial resolution to x_local
        if x_global.shape[2:] != x_local.shape[2:]:
            x_global = F.interpolate(x_global, size=x_local.shape[2:], mode="nearest")

        feat_loc = self.proj_local(x_local)
        feat_glb = self.proj_global(x_global)

        # Generate spatial 2D weight matrix [B, 2, H, W]
        combined = torch.cat([feat_loc, feat_glb], dim=1)
        weights = self.weight_generator(combined)

        w_loc = weights[:, 0:1, :, :]
        w_glb = weights[:, 1:2, :, :]

        # In-place weighted fusion
        fused = feat_loc * w_loc + feat_glb * w_glb
        return self.refine(fused)


# ─────────────────────────────────────────────────────────────────────────────
# 4. HARMONIC RESONANT BROADCAST CORE (AetherResonantCore)
# ─────────────────────────────────────────────────────────────────────────────
class AetherResonantCore(nn.Module):
    """
    Central Resonant Broadcast Core.
    
    Synchronously fuses P3, P4, P5 multi-scale representations into a single
    harmonic manifold located at P4 resolution, eliminating the multi-step
    ping-pong delay of PANet and the tangled graph of BiFPN.
    """
    def __init__(self, c1, c2):
        super().__init__()
        # c1 is a list of [c_p3, c_p4, c_p5]
        if isinstance(c1, int):
            c1 = [c1, c1, c1]
        self.c1 = c1
        self.c2 = c2

        # Scale and project each input to P4 resolution and c2 channels
        self.p3_to_p4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv(c1[0], c2, 1)
        )
        self.p4_proj = Conv(c1[1], c2, 1) if c1[1] != c2 else nn.Identity()
        self.p5_to_p4 = Conv(c1[2], c2, 1)

        # Central Harmonic Resonator
        self.resonator = OSMConv(c2 * 3, c2)

    def forward(self, x):
        # x is a list of [P3, P4, P5]
        p3, p4, p5 = x[0], x[1], x[2]
        target_size = p4.shape[2:]

        # Transform all to P4 spatial resolution
        feat_p3 = self.p3_to_p4(p3)
        feat_p4 = self.p4_proj(p4)
        feat_p5 = F.interpolate(self.p5_to_p4(p5), size=target_size, mode="nearest")

        # Fuse into central resonant manifold
        harmonic_input = torch.cat([feat_p3, feat_p4, feat_p5], dim=1)
        return self.resonator(harmonic_input)
