"""Checkpoint-compatible execution optimization for Orthogonal scale fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .orthogonal_blocks import OSIFusion

__all__ = ("OSIFusionEfficient",)


class OSIFusionEfficient(OSIFusion):
    """Project coarse features before nearest upsampling during evaluation.

    Input: two BCHW tensors [local, global]; output: B x c_out x H_local x W_local.
    Parameters: inherited c_local, c_global, c_out and the same state_dict as OSIFusion.
    Purpose: avoid evaluating a pointwise projection on replicated coarse pixels.
    Training retains the original path, including BatchNorm statistics on odd shapes.
    Reference: pointwise lateral projection in FPN and weighted fusion in BiFPN.
    This is an execution optimization, not a claim of a new fusion mechanism.
    """

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        local, coarse = x
        target = local.shape[2:]
        source = coarse.shape[2:]
        if self.training or source[0] > target[0] or source[1] > target[1]:
            return super().forward(x)

        local = self.proj_loc(local)
        coarse = self.proj_glb(coarse)
        if source != target:
            coarse = F.interpolate(coarse, size=target, mode="nearest")
        positive = F.relu(self.w)
        weight = positive / (positive.sum() + self.epsilon)
        return self.act(weight[0] * local + weight[1] * coarse)
