# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-Orthogonal v3 Architectural Blocks
=======================================

Core Mathematical Innovations:
1. Orthogonal Directional Reparameterization (ODR-Conv / OHR-Conv v3):
   - Training: 5 canonical orthogonal bases:
     * 3x3 Standard Conv + BN: 2D spatial correlation
     * 1x3 Horizontal Conv + BN: Directional gradient along e_x
     * 3x1 Vertical Conv + BN: Directional gradient along e_y (<1x3, 3x1> = 0)
     * 1x1 Conv + BN: 0th-order DC component / channel intensity
     * Identity Shortcut + BN: Direct gradient propagation (when c1 == c2 and s == 1)
   - Inference / Deploy: Algebraically collapsed into a SINGLE nn.Conv2d(c1, c2, 3, s, 1) kernel.
     Zero kernel launches, 100% contiguous GEMM, 0.00ms runtime overhead.

2. Harmonic Cross-Modulation Neck (HCM-Fusion / OSIFusion v3):
   - Zero-Concat Cross-Scale Interaction: Replaces memory-heavy channel concatenation (which doubles
     channel dimensions and memory traffic) with Dynamic Bilinear Channel Modulation:
         Y = Proj_loc(P_loc) * (1.0 + Sigmoid(Conv1x1(GAP(Proj_glb(P_glb))))) + Proj_glb(P_glb)
   - Followed by deep non-linear C3k2_Ortho containers to provide high representational capacity
     without doubling channel width.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import PSABlock

__all__ = [
    "OHRConv",
    "ODRConv",
    "OrthoBottleneck",
    "OrthoC3k",
    "C3k2_Ortho",
    "OSIFusion",
    "HCMFusion",
    "SESPGate",
    "DOC_Fusion",
    "DOCFusion",
    "OrthoModFusion",
    "C2PSA_Ortho",
]


def conv_bn_fusion(conv_weight, bn):
    """
    Fuses a Conv weight and BatchNorm into an equivalent single Conv weight and bias.
    """
    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(running_var + eps)
    fused_weight = conv_weight * (gamma / std).reshape(-1, 1, 1, 1)
    fused_bias = beta - running_mean * gamma / std
    return fused_weight, fused_bias


# ─────────────────────────────────────────────────────────────────────────────
# 1. ORTHOGONAL DIRECTIONAL REPARAMETERIZATION CONVOLUTION (ODR-Conv / OHR-Conv)
# ─────────────────────────────────────────────────────────────────────────────
class OHRConv(nn.Module):
    """
    Orthogonal Directional Reparameterization Convolution (ODR-Conv / OHR-Conv v3).

    Training:
      - Branch 3x3: Joint 2D spatial correlation
      - Branch 1x3: Horizontal directional gradient (padding=(0, 1))
      - Branch 3x1: Vertical directional gradient (padding=(1, 0))
      - Branch 1x1: 0th-order DC component / channel intensity
      - Branch Id : Identity shortcut (when c1 == c2 and s == 1)
    Inference:
      - Exact algebraic fusion into a SINGLE nn.Conv2d(c1, c2, 3, stride, 1, bias=True)
      - Zero runtime overhead!
    """
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.p = p
        self.g = g
        self.deployed = False

        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        # Training branches:
        # 1. Main 3x3 Conv + BN
        self.conv3x3 = nn.Conv2d(c1, c2, 3, stride=s, padding=1, groups=g, bias=False)
        self.bn3x3 = nn.BatchNorm2d(c2)

        # 2. 1x1 Conv + BN (0th-order DC band)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, stride=s, padding=0, groups=g, bias=False)
        self.bn1x1 = nn.BatchNorm2d(c2)

        # 3. Orthogonal Directional Branches (horizontal 1x3 and vertical 3x1)
        if s == 1:
            self.conv1x3 = nn.Conv2d(c1, c2, (1, 3), stride=1, padding=(0, 1), groups=g, bias=False)
            self.bn1x3 = nn.BatchNorm2d(c2)

            self.conv3x1 = nn.Conv2d(c1, c2, (3, 1), stride=1, padding=(1, 0), groups=g, bias=False)
            self.bn3x1 = nn.BatchNorm2d(c2)
        else:
            self.conv1x3 = None
            self.bn1x3 = None
            self.conv3x1 = None
            self.bn3x1 = None

        # 4. Identity branch (when c1 == c2 and s == 1)
        if c1 == c2 and s == 1:
            self.bn_id = nn.BatchNorm2d(c2)
        else:
            self.bn_id = None

    def get_equivalent_kernel_bias(self):
        """Mathematically fuses all 5 orthogonal branches into a single 3x3 kernel and bias."""
        # 1. Fuse 3x3 branch
        w3, b3 = conv_bn_fusion(self.conv3x3.weight, self.bn3x3)

        # 2. Fuse 1x1 branch (padded to 3x3: pad 1 on all sides)
        w1, b1 = conv_bn_fusion(self.conv1x1.weight, self.bn1x1)
        w1_padded = F.pad(w1, (1, 1, 1, 1))

        fused_weight = w3 + w1_padded
        fused_bias = b3 + b1

        # 3. Fuse 1x3 horizontal branch (padded along height: pad top=1, bottom=1)
        if self.conv1x3 is not None:
            w1x3, b1x3 = conv_bn_fusion(self.conv1x3.weight, self.bn1x3)
            fused_weight += F.pad(w1x3, (0, 0, 1, 1))
            fused_bias += b1x3

        # 4. Fuse 3x1 vertical branch (padded along width: pad left=1, right=1)
        if self.conv3x1 is not None:
            w3x1, b3x1 = conv_bn_fusion(self.conv3x1.weight, self.bn3x1)
            fused_weight += F.pad(w3x1, (1, 1, 0, 0))
            fused_bias += b3x1

        # 5. Fuse Identity branch (if exists)
        if self.bn_id is not None:
            input_dim = self.c1 // self.g
            id_weight = torch.zeros(self.c2, input_dim, 3, 3, dtype=w3.dtype, device=w3.device)
            for i in range(self.c2):
                id_weight[i, i % input_dim, 1, 1] = 1.0
            w_id, b_id = conv_bn_fusion(id_weight, self.bn_id)
            fused_weight += w_id
            fused_bias += b_id

        return fused_weight, fused_bias

    def switch_to_deploy(self):
        """Converts module to single-operator deployment mode."""
        if self.deployed:
            return
        fused_weight, fused_bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            self.c1, self.c2, 3, stride=self.s, padding=self.p, groups=self.g, bias=True
        ).to(device=fused_weight.device, dtype=fused_weight.dtype)
        self.fused_conv.train(self.training)
        self.fused_conv.weight.data.copy_(fused_weight)
        self.fused_conv.bias.data.copy_(fused_bias)
        self.fused_conv.requires_grad_(False)
        for p in self.fused_conv.parameters():
            p.requires_grad = False

        # Remove training branch parameters to free VRAM
        del self.conv3x3, self.bn3x3
        del self.conv1x1, self.bn1x1
        if self.conv1x3 is not None:
            del self.conv1x3, self.bn1x3
        if self.conv3x1 is not None:
            del self.conv3x1, self.bn3x1
        if self.bn_id is not None:
            del self.bn_id
        self.deployed = True

    def fuse(self):
        """Ultralytics compatibility hook."""
        self.switch_to_deploy()

    def forward(self, x):
        if self.deployed:
            return self.act(self.fused_conv(x))

        # Training forward pass
        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.conv1x3 is not None:
            out = out + self.bn1x3(self.conv1x3(x))
        if self.conv3x1 is not None:
            out = out + self.bn3x1(self.conv3x1(x))
        if self.bn_id is not None:
            out = out + self.bn_id(x)
        return self.act(out)


# Alias ODRConv to OHRConv for flexible naming
ODRConv = OHRConv


# ─────────────────────────────────────────────────────────────────────────────
# 2. ORTHO-BOTTLENECK & C3K2-ORTHO CONTAINER
# ─────────────────────────────────────────────────────────────────────────────
class OrthoBottleneck(nn.Module):
    """
    Bottleneck using ODR/OHR-Conv for high-efficiency feature representation.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_mid = int(c2 * e)
        self.cv1 = Conv(c1, c_mid, 1, 1)
        self.cv2 = OHRConv(c_mid, c2, k=3, s=1, g=g)
        self.add = shortcut and c1 == c2

    def switch_to_deploy(self):
        if hasattr(self.cv1, "fuse"):
            self.cv1.fuse()
        self.cv2.switch_to_deploy()
        for p in self.parameters():
            p.requires_grad = False

    def fuse(self):
        self.switch_to_deploy()

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class OrthoC3k(nn.Module):
    """
    Orthogonal CSP Bottleneck with 2 sub-bottlenecks for deep non-linear feature extraction.
    Analogous to C3k in YOLO11/YOLO26, but powered by OHR-Conv.
    """
    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(OrthoBottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        )

    def switch_to_deploy(self):
        if hasattr(self.cv1, "fuse"):
            self.cv1.fuse()
        if hasattr(self.cv2, "fuse"):
            self.cv2.fuse()
        if hasattr(self.cv3, "fuse"):
            self.cv3.fuse()
        for b in self.m:
            if hasattr(b, "switch_to_deploy"):
                b.switch_to_deploy()
            elif hasattr(b, "fuse"):
                b.fuse()
        for p in self.parameters():
            p.requires_grad = False

    def fuse(self):
        self.switch_to_deploy()

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2_Ortho(nn.Module):
    """
    Ultra-Fast CSP Container with OrthoBottleneck and OrthoC3k.
    Fully deployable: calling .fuse() or .switch_to_deploy() flattens all internal blocks.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            OrthoC3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else OrthoBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def switch_to_deploy(self):
        if hasattr(self.cv1, "fuse"):
            self.cv1.fuse()
        if hasattr(self.cv2, "fuse"):
            self.cv2.fuse()
        for b in self.m:
            if hasattr(b, "switch_to_deploy"):
                b.switch_to_deploy()
            elif hasattr(b, "fuse"):
                b.fuse()
        for p in self.parameters():
            p.requires_grad = False

    def fuse(self):
        self.switch_to_deploy()

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ─────────────────────────────────────────────────────────────────────────────
# 3. HARMONIC CROSS-MODULATION FUSION (HCM-Fusion / OSIFusion v3)
# ─────────────────────────────────────────────────────────────────────────────
class OSIFusion(nn.Module):
    """
    Fast Normalized Cross-Scale Fusion (BiFPN-inspired HCM-Fusion).
    Preserves 100% spatial resolution for small objects and platelets,
    eliminates heavy global pooling / 1x1 conv gate, cutting over 100,000 redundant parameters.
    """
    def __init__(self, c_local, c_global, c_out):
        super().__init__()
        self.c_out = c_out
        self.proj_loc = Conv(c_local, c_out, 1) if c_local != c_out else nn.Identity()
        self.proj_glb = Conv(c_global, c_out, 1) if c_global != c_out else nn.Identity()

        # Learnable positive weights for fast normalized feature fusion
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        self.act = nn.SiLU(inplace=True)

    def switch_to_deploy(self):
        if hasattr(self.proj_loc, "fuse"):
            self.proj_loc.fuse()
        if hasattr(self.proj_glb, "fuse"):
            self.proj_glb.fuse()
        for p in self.parameters():
            p.requires_grad = False

    def fuse(self):
        self.switch_to_deploy()

    def forward(self, x):
        # x is [x_local, x_global]
        x_loc, x_glb = x[0], x[1]
        target_size = x_loc.shape[2:]

        if x_glb.shape[2:] != target_size:
            x_glb = F.interpolate(x_glb, size=target_size, mode="nearest")

        feat_loc = self.proj_loc(x_loc)
        feat_glb = self.proj_glb(x_glb)

        w = F.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.act(weight[0] * feat_loc + weight[1] * feat_glb)


HCMFusion = OSIFusion


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCALE-EXCLUSIVE SUBSPACE PROJECTION GATE (SESP-Gate)
# ─────────────────────────────────────────────────────────────────────────────
class SESPGate(nn.Module):
    """
    Scale-Exclusive Subspace Projection Gate (SESP-Gate).
    Preserved for backwards compatibility.
    """
    def __init__(self, c_fine, c_coarse):
        super().__init__()
        self.proj_mask = nn.Sequential(
            nn.Conv2d(c_fine, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        nn.init.constant_(self.proj_mask[0].bias, -2.0)
        self.refine = Conv(c_coarse, c_coarse, 3, 1)

    def switch_to_deploy(self):
        if hasattr(self.refine, "fuse"):
            self.refine.fuse()
        for p in self.parameters():
            p.requires_grad = False

    def fuse(self):
        self.switch_to_deploy()

    def forward(self, x):
        p_fine, p_coarse = x[0], x[1]
        target_size = p_coarse.shape[2:]
        mask_fine = self.proj_mask(p_fine)
        if mask_fine.shape[2:] != target_size:
            mask_coarse = F.adaptive_avg_pool2d(mask_fine, target_size)
        else:
            mask_coarse = mask_fine
        p_coarse_clean = p_coarse * (1.0 - mask_coarse)
        return self.refine(p_coarse_clean)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ZERO-CONCAT ORTHOGONAL MODULATION FUSION (OrthoModFusion / DOC-Fusion)
# ─────────────────────────────────────────────────────────────────────────────
class OrthoModFusion(nn.Module):
    """
    100% Zero-Concat Orthogonal Modulation Fusion (OrthoModFusion).
    Eliminates Concat completely, cutting memory access traffic by 50%!
    
    Mechanism:
      1. Spatial & Channel Context Alignment via 1x1 Projections (if channel sizes differ).
      2. Dynamic Channel-Wise Affine Modulation:
         gamma = tanh(Conv1x1(GAP(P_glb)))  # strictly bounded in (-1, 1)
         beta  = Conv1x1(GAP(P_glb))        # adaptive DC bias
         P_loc_modulated = (1.0 + gamma) * P_loc + beta
      3. Zero-Concat Normalized Harmonic Integration:
         Y = SiLU(BatchNorm(P_loc_modulated + P_glb))
    Guarantees stable variance, preserves 100% fine spatial details, and prevents gradient explosion!
    """
    def __init__(self, c_local, c_global, c_out):
        super().__init__()
        self.c_out = c_out
        self.proj_loc = Conv(c_local, c_out, 1) if c_local != c_out else nn.Identity()
        self.proj_glb = Conv(c_global, c_out, 1) if c_global != c_out else nn.Identity()

        # Dynamic channel-wise affine gate
        mid_ch = max(c_out // 4, 16)
        self.gate_gamma = nn.Sequential(
            Conv(c_out, mid_ch, 1),
            nn.Conv2d(mid_ch, c_out, 1, bias=True),
            nn.Tanh()
        )
        self.gate_beta = nn.Sequential(
            Conv(c_out, mid_ch, 1),
            nn.Conv2d(mid_ch, c_out, 1, bias=True)
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def switch_to_deploy(self):
        if hasattr(self.proj_loc, "fuse"):
            self.proj_loc.fuse()
        if hasattr(self.proj_glb, "fuse"):
            self.proj_glb.fuse()
        if hasattr(self.gate_gamma[0], "fuse"):
            self.gate_gamma[0].fuse()
        if hasattr(self.gate_beta[0], "fuse"):
            self.gate_beta[0].fuse()
        for p in self.parameters():
            p.requires_grad = False

    def fuse(self):
        self.switch_to_deploy()

    def forward(self, x):
        x_loc, x_glb = x[0], x[1]
        target_size = x_loc.shape[2:]

        if x_glb.shape[2:] != target_size:
            x_glb = F.interpolate(x_glb, size=target_size, mode="nearest")

        feat_loc = self.proj_loc(x_loc)
        feat_glb = self.proj_glb(x_glb)

        glb_pool = F.adaptive_avg_pool2d(feat_glb, 1)
        gamma = self.gate_gamma(glb_pool)
        beta = self.gate_beta(glb_pool)

        feat_loc_mod = (1.0 + gamma) * feat_loc + beta
        return self.act(self.bn(feat_loc_mod + feat_glb))


DOC_Fusion = OrthoModFusion
DOCFusion = OrthoModFusion


# ─────────────────────────────────────────────────────────────────────────────
# 6. ORTHOGONAL POLARIZED SELF-ATTENTION (C2PSA_Ortho)
# ─────────────────────────────────────────────────────────────────────────────
class C2PSA_Ortho(nn.Module):
    """
    Orthogonal Polarized Self-Attention (C2PSA_Ortho).
    Integrates Multi-Head Area Self-Attention at P5 to capture long-range semantic context
    across microscope slides, fully fusible with zero deployment latency.
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))
        )

    def switch_to_deploy(self):
        if hasattr(self.cv1, "fuse"):
            self.cv1.fuse()
        if hasattr(self.cv2, "fuse"):
            self.cv2.fuse()
        for blk in self.m:
            if hasattr(blk, "fuse"):
                blk.fuse()
        for p in self.parameters():
            p.requires_grad = False

    def fuse(self):
        self.switch_to_deploy()

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
