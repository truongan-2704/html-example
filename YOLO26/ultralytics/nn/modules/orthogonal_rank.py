# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Center-preserving spatial rank restriction for Orthogonal architecture experiments."""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv
from .head import Detect
from .orthogonal_blocks import C3k2_Ortho, OHRConv, OrthoBottleneck

__all__ = ("CenterRankConv", "C3k2_OrthoRank", "OrthoRankDetect", "transfer_orthogonal_rank")


class CenterRankConv(nn.Module):
    """Mix all center channels and a narrow spatial response in one output projection.

    Input: BCHW with c1 channels. Output: B,c2,ceil(H/s),ceil(W/s).
    Parameters: c1/c2 channels; k=3; s=1/2; ratio in (0,1] controls spatial rank.
    Purpose: restrict neighboring-pixel channel interaction while leaving center
        channel mixing unrestricted. Deploy uses a narrow 3x3 and a 1x1 mixer.
    References: low-rank CNN filters and ACNet/RepVGG-style OHR reparameterization.
        This local architecture candidate has no established originality claim.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, ratio: float = 0.25, act: bool = True):
        super().__init__()
        if c1 < 1 or c2 < 1 or k != 3 or s not in (1, 2) or not 0 < ratio <= 1:
            raise ValueError("CenterRankConv requires positive channels, k=3, s=1/2 and 0 < ratio <= 1")
        self.c1, self.c2, self.s = c1, c2, s
        self.rank = min(min(c1, c2), max(1, math.ceil(min(c1, c2) * ratio / 8) * 8))
        self.spatial = OHRConv(c1, self.rank, s=s, act=False)
        self.mix = Conv(c1 + self.rank, c2, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate center samples and learned low-rank spatial responses."""
        spatial = self.spatial(x)
        center = x if self.s == 1 else x[:, :, ::self.s, ::self.s]
        return self.mix(torch.cat((center, spatial), dim=1))

    def switch_to_deploy(self) -> None:
        """Fuse OHR branches and output BN without expanding to a dense spatial kernel."""
        self.spatial.switch_to_deploy()
        if hasattr(self.mix, "bn"):
            self.mix.conv = fuse_conv_and_bn(self.mix.conv, self.mix.bn)
            del self.mix.bn
            self.mix.forward = self.mix.forward_fuse

    @torch.no_grad()
    def initialize_from_kernel(self, weight: torch.Tensor, bias: torch.Tensor) -> dict:
        """Approximate an eval 3x3 kernel by rank-r off-center SVD; preserve its center.

        Requires an unfused target. Returns relative off-center reconstruction error.
        This is approximate weight initialization, not accuracy-preserving conversion.
        """
        if self.spatial.deployed or not hasattr(self.mix, "bn"):
            raise ValueError("Initialize an unfused CenterRankConv before deployment")
        if tuple(weight.shape) != (self.c2, self.c1, 3, 3) or tuple(bias.shape) != (self.c2,):
            raise ValueError("Source kernel/bias dimensions do not match this module")
        original = weight.detach().double().cpu()
        flat = original.reshape(self.c2, self.c1, 9)
        offsets = [0, 1, 2, 3, 5, 6, 7, 8]
        matrix = flat[:, :, offsets].reshape(self.c2, -1)
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        left = u[:, :self.rank] * singular[:self.rank]
        right = vh[:self.rank]
        spatial = torch.zeros(self.rank, self.c1, 9, dtype=torch.float64)
        spatial[:, :, offsets] = right.reshape(self.rank, self.c1, 8)

        # Every branch is initially zero except the main spatial basis.
        for module in self.spatial.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.zero_()
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.zero_()
                module.running_var.fill_(1)
                module.weight.fill_(math.sqrt(1 + module.eps))
                module.bias.zero_()
                module.num_batches_tracked.zero_()
        # Equal channel count creates an OHR identity branch; suppress it at initialization.
        if self.spatial.bn_id is not None:
            self.spatial.bn_id.weight.zero_()
        self.spatial.conv3x3.weight.copy_(spatial.reshape(self.rank, self.c1, 3, 3))
        mix = torch.cat((original[:, :, 1, 1], left), dim=1)
        self.mix.conv.weight.copy_(mix[:, :, None, None])
        self.mix.bn.running_mean.zero_()
        self.mix.bn.running_var.fill_(1)
        self.mix.bn.weight.fill_(math.sqrt(1 + self.mix.bn.eps))
        self.mix.bn.bias.copy_(bias)
        self.mix.bn.num_batches_tracked.zero_()
        error = torch.linalg.vector_norm(matrix - left @ right)
        denom = torch.linalg.vector_norm(matrix)
        return dict(rank=self.rank, relative_offcenter_error=float(error / denom) if denom > 0 else 0.0)


class C3k2_OrthoRank(C3k2_Ortho):
    """Existing CSP paths with CenterRankConv replacing internal OHR spatial kernels.

    Input/output and c1,c2,n,c3k,e,g,shortcut match C3k2_Ortho; ratio sets spatial rank.
    Purpose: isolate spatial-rank restriction from CSP width/repeat changes.
    References and originality limitations: CenterRankConv. Only g=1 is supported.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, ratio=0.25):
        if g != 1:
            raise ValueError("C3k2_OrthoRank currently supports g=1 only")
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        for block in list(self.modules()):
            if isinstance(block, OrthoBottleneck):
                old = block.cv2
                block.cv2 = CenterRankConv(old.c1, old.c2, ratio=ratio)


class OrthoRankDetect(Detect):
    """Detect with center-preserving rank restriction in both regression convolutions.

    Input/output follow Detect exactly. nc, reg_max, end2end and ch retain their
    meanings; ratio sets the spatial rank. Classification, decoding and loss are
    inherited unchanged. No sharing between scales or one-to-many/one-to-one heads.
    Purpose: target measured dense regression cost without narrowing its outputs.
    References and originality limitations: CenterRankConv.
    """

    def __init__(self, nc=80, ratio=0.25, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        width = max(16, ch[0] // 4, reg_max * 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(CenterRankConv(c, width, ratio=ratio), CenterRankConv(width, width, ratio=ratio),
                          nn.Conv2d(width, 4 * reg_max, 1)) for c in ch
        )
        if end2end:
            self.one2one_cv2 = deepcopy(self.cv2)


@torch.no_grad()
def transfer_orthogonal_rank(source: nn.Module, target: nn.Module) -> dict:
    """Copy unaffected tensors and approximate changed Conv/OHR kernels by SVD.

    Inputs are matching-scale unfused DetectionModels with the same class count.
    The source is never mutated. Target is initialized in place. Report every
    changed layer; this is not a strict or lossless pretrained-weight conversion.
    """
    source_state, target_state = source.state_dict(), target.state_dict()
    matched = {k: v for k, v in source_state.items() if k in target_state and v.shape == target_state[k].shape}
    # Validate all changed module correspondences before any writes.
    replacements = []
    source_modules = dict(source.named_modules())
    for name, module in target.named_modules():
        if isinstance(module, CenterRankConv):
            if module.spatial.deployed or not hasattr(module.mix, "bn"):
                raise ValueError("Weight transfer requires an unfused target architecture")
            old = source_modules.get(name)
            if not isinstance(old, (Conv, OHRConv)):
                raise ValueError(f"No compatible source Conv/OHR at {name}; use the matching original architecture")
            if isinstance(old, OHRConv):
                weight, bias = old.get_equivalent_kernel_bias() if not old.deployed else (old.fused_conv.weight, old.fused_conv.bias)
            else:
                # This vendored fuse helper mutates its Conv argument in place.
                fused = fuse_conv_and_bn(deepcopy(old.conv), old.bn) if hasattr(old, "bn") else old.conv
                weight, bias = fused.weight, fused.bias
            if tuple(weight.shape) != (module.c2, module.c1, 3, 3):
                raise ValueError(f"Channel mismatch at {name}; source and target need the same scale")
            replacements.append((name, module, weight, bias))
    if source.model[-1].nc != target.model[-1].nc or source.model[-1].end2end != target.model[-1].end2end:
        raise ValueError("Source and target must have matching classes and end2end settings")
    target.load_state_dict(matched, strict=False)
    report = {name: module.initialize_from_kernel(weight, bias) for name, module, weight, bias in replacements}
    return dict(copied_tensors=len(matched), approximated_layers=report, exact_checkpoint_transfer=False)
