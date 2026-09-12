"""Numerical and integration regressions for opt-in selected decoding."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research'))

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.selected_decode import SelectDecodeDetect
from test_orthogonal_efficient import build, CONFIG_DIR


def raw_predictions(nc=4, reg_max=16, batch=2, dtype=torch.float32):
    feats = [torch.randn(batch, c, h, w, dtype=dtype) for c, h, w in ((8, 16, 20), (16, 8, 10), (32, 4, 5))]
    positions = sum(v.shape[-2] * v.shape[-1] for v in feats)
    return dict(feats=feats, boxes=torch.randn(batch, 4 * reg_max, positions, dtype=dtype) * 2,
                scores=torch.randn(batch, nc, positions, dtype=dtype) * 2)


class SelectedDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(102)

    def test_decode_oracle_classes_bins_ties_batch_and_limits(self):
        for nc in (1, 4, 80):
            for reg_max in (1, 16):
                for dtype in (torch.float32, torch.float64):
                    m = SelectDecodeDetect(nc, reg_max, True, (8, 16, 32)).to(dtype=dtype).eval()
                    m.stride = torch.tensor([8., 16., 32.], dtype=dtype)
                    preds = raw_predictions(nc, reg_max, dtype=dtype)
                    original_scores = preds['scores'].clone()
                    for agnostic in (False, True):
                        for k in (7, 1000):
                            for ties in (False, True):
                                with self.subTest(nc=nc, bins=reg_max, dtype=dtype, agnostic=agnostic, k=k, ties=ties):
                                    m.agnostic_nms, m.max_det = agnostic, k
                                    preds['scores'] = original_scores.clone()
                                    if ties:
                                        preds['scores'].zero_()
                                        # Saturated scores and tied classes must retain original selection.
                                        preds['scores'][:, :, :2] = 100
                                    with torch.inference_mode():
                                        expected = m.postprocess(m._inference(preds).permute(0, 2, 1))
                                        actual = m._selected_inference(preds)
                                    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
                                    torch.testing.assert_close(actual[..., 4:], expected[..., 4:], rtol=0, atol=0)
                                    self.assertTrue(torch.isfinite(actual).all())

    def test_dfl_receives_only_selected_positions_and_preserves_duplicates(self):
        m = SelectDecodeDetect(4, 16, True, (8, 16, 32)).eval()
        m.stride = torch.tensor([8., 16., 32.])
        m.max_det = 3
        preds = raw_predictions()
        preds['scores'].fill_(-20)
        preds['scores'][:, :3, 0] = torch.tensor([8., 7., 6.])
        seen = []
        hook = m.dfl.register_forward_pre_hook(lambda module, x: seen.append(tuple(x[0].shape)))
        with torch.inference_mode():
            output = m._selected_inference(preds)
        hook.remove()
        self.assertEqual(seen, [(2, 64, 3)])
        torch.testing.assert_close(output[:, 0, :4], output[:, 1, :4], rtol=0, atol=0)
        torch.testing.assert_close(output[:, 1, :4], output[:, 2, :4], rtol=0, atol=0)
        torch.testing.assert_close(output[..., 5], torch.tensor([[0., 1., 2.], [0., 1., 2.]]))

    def test_non_end2end_delegates(self):
        old = Detect(4, 16, False, (8, 16, 32)).eval()
        new = SelectDecodeDetect(4, 16, False, (8, 16, 32)).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        old.stride = new.stride = torch.tensor([8., 16., 32.])
        x = raw_predictions()['feats']
        with torch.inference_mode():
            torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)

    def test_full_model_training_loss_gradients_and_eval(self):
        old = build('yolo11-orthogonal.yaml')
        new = build('yolo11-orthogonal-efficient-sd.yaml')
        new.load_state_dict(old.state_dict(), strict=True)
        batch = dict(img=torch.randn(2, 3, 64, 96), batch_idx=torch.tensor([0., 1.]),
                     cls=torch.tensor([[0.], [1.]]), bboxes=torch.tensor([[.5, .5, .4, .4], [.4, .4, .3, .3]]))
        losses = []
        for model in (old, new):
            model.train()
            model.args = get_cfg()
            loss, _ = model(batch)
            loss.sum().backward()
            losses.append(loss)
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for p, q in zip(old.parameters(), new.parameters()):
            if p.requires_grad:
                self.assertIsNotNone(q.grad)
                self.assertTrue(torch.isfinite(q.grad).all())
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        torch.testing.assert_close(old.state_dict(), new.state_dict(), rtol=0, atol=0)
        for fused in (False, True):
            old.eval()
            new.eval()
            if fused:
                old.fuse(verbose=False)
                new.fuse(verbose=False)
            for shape in ((1, 3, 128, 160), (2, 3, 96, 64), (2, 3, 32, 32)):
                x = torch.randn(*shape)
                with torch.inference_mode():
                    a, b = old(x), new(x)
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
                self.assertTrue(torch.isfinite(b[0]).all())

    def test_parser_scales_and_api_checkpoint(self):
        for config in ('yolo11-orthogonal-sd.yaml', 'yolo11-orthogonal-efficient-sd.yaml'):
            for scale in 'nsmlx':
                with self.subTest(config=config, scale=scale):
                    model = build(config, scale=scale)
                    self.assertIsInstance(model.model[-1], SelectDecodeDetect)
                    self.assertTrue(model.end2end)
                    self.assertEqual(model.stride.tolist(), [8., 16., 32.])
                    del model
        model = YOLO(str(CONFIG_DIR / 'yolo11-orthogonal-efficient-sd.yaml'))
        with tempfile.TemporaryDirectory(dir=ROOT / 'experiments') as tmp:
            path = Path(tmp) / 'selected.pt'
            model.save(path)
            loaded = YOLO(path).model.eval()
            self.assertIsInstance(loaded.model[-1], SelectDecodeDetect)
            expected = deepcopy(model.model).half().float().eval()
            x = torch.randn(2, 3, 64, 96)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(x), expected(x), rtol=0, atol=0)

    def test_cpu_autocast(self):
        old = build('yolo11-orthogonal.yaml').eval().fuse(verbose=False)
        new = build('yolo11-orthogonal-sd.yaml').eval().fuse(verbose=False)
        new.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(2, 3, 64, 96)
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            a, b = old(x), new(x)
        torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)
        self.assertTrue(torch.isfinite(b[0]).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable in installed PyTorch build')
    def test_cuda_amp_selected_decoding(self):
        m = SelectDecodeDetect(4, 16, True, (8, 16, 32)).cuda().eval()
        m.stride = torch.tensor([8., 16., 32.], device='cuda')
        p = raw_predictions()
        p = {k: [v.cuda() for v in value] if k == 'feats' else value.cuda() for k, value in p.items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            expected = m.postprocess(m._inference(p).permute(0, 2, 1))
            actual = m._selected_inference(p)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
