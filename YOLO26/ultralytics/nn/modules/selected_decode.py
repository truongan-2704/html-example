# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Opt-in end-to-end head that decodes only positions selected by class scores."""

from __future__ import annotations

import torch

from ultralytics.utils.tal import make_anchors

from .head import Detect

__all__ = ("SelectDecodeDetect",)


class SelectDecodeDetect(Detect):
    """Preserve Detect training and weights while gathering distributions before DFL.

    Input: the same list of BCHW feature maps as Detect.
    Output: original training dictionaries, or B x min(max_det, N) x 6 detections
        with the original raw-prediction tuple when not exporting.
    Parameters: inherited nc, reg_max, end2end and ch; no additional parameters.
    Purpose: reduce position-independent decoding work in end-to-end evaluation.
    Reference: pre-decode candidate filtering in MMDetection and D-Robotics YOLO
        conversion. This execution ordering is not claimed as a new mechanism.
    Export retains Detect's requirement that N >= max_det.
    """

    def forward(self, x: list[torch.Tensor]) -> dict | torch.Tensor | tuple:
        """Keep training untouched; select and decode the one-to-one eval predictions."""
        if self.training or not self.end2end:
            return super().forward(x)

        one2many = self.forward_head(x, **self.one2many)
        one2one = self.forward_head([xi.detach() for xi in x], **self.one2one)
        y = self._selected_inference(one2one)
        return y if self.export else (y, {"one2many": one2many, "one2one": one2one})

    def _selected_inference(self, preds: dict) -> torch.Tensor:
        """Gather raw distributions, anchors and strides using the original top-k rule."""
        shape = preds["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1) for a in make_anchors(preds["feats"], self.stride, 0.5)
            )
            self.shape = shape

        scores, labels, indices = self.get_topk_index(preds["scores"].sigmoid().permute(0, 2, 1), self.max_det)
        index = indices.transpose(1, 2)
        batch = preds["boxes"].shape[0]
        distributions = preds["boxes"].gather(2, index.expand(-1, 4 * self.reg_max, -1))
        anchors = self.anchors.unsqueeze(0).expand(batch, -1, -1).gather(2, index.expand(-1, 2, -1))
        strides = self.strides.unsqueeze(0).expand(batch, -1, -1).gather(2, index)
        boxes = self.decode_bboxes(self.dfl(distributions), anchors) * strides
        return torch.cat((boxes.transpose(1, 2), scores, labels), dim=-1)
