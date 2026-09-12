"""Architecture, SVD initialization, loss, deployment and baseline regression tests."""
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
from ultralytics.nn.modules.orthogonal_rank import CenterRankConv, OrthoRankDetect, transfer_orthogonal_rank
from ultralytics.nn.tasks import DetectionModel
from orthogonal_rank_common import variant_config, VARIANTS, CONFIGS
from test_orthogonal_efficient import random_bn


def build(variant='bc', ratio=0.25, scale='n', nc=4):
    return DetectionModel(variant_config(variant, ratio, scale, nc), verbose=False)


class OrthogonalRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(94)

    def test_shapes_backward_odd_even_stride_and_invalid_args(self):
        for c1, c2 in ((3, 7), (16, 32), (32, 16), (64, 64)):
            for stride in (1, 2):
                for h, w in ((9, 13), (8, 12)):
                    with self.subTest(channels=(c1, c2), stride=stride, shape=(h, w)):
                        m = CenterRankConv(c1, c2, s=stride)
                        x = torch.randn(2, c1, h, w, requires_grad=True)
                        before = x.detach().clone()
                        y = m(x)
                        self.assertEqual(y.shape, (2, c2, (h + stride - 1)//stride, (w + stride - 1)//stride))
                        y.square().mean().backward()
                        self.assertTrue(torch.isfinite(y).all() and torch.isfinite(x.grad).all())
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                        torch.testing.assert_close(x.detach(), before, rtol=0, atol=0)
        for kwargs in (dict(k=5), dict(s=3), dict(ratio=0), dict(ratio=1.1)):
            with self.assertRaises(ValueError):
                CenterRankConv(16, 16, **kwargs)

    def test_deploy_equivalence_nontrivial_bn_dtype_stride(self):
        for dtype in (torch.float32, torch.float64):
            for stride in (1, 2):
                m = CenterRankConv(32, 48, s=stride).to(dtype=dtype).eval()
                random_bn(m)
                x = torch.randn(2, 32, 13, 10, dtype=dtype)
                with torch.inference_mode():
                    expected = m(x)
                    m.switch_to_deploy()
                    actual = m(x)
                    m.switch_to_deploy()
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                self.assertEqual(m.spatial.fused_conv.weight.dtype, dtype)
                self.assertEqual(m.mix.conv.weight.dtype, dtype)
                self.assertEqual(sum(isinstance(b, torch.nn.BatchNorm2d) for b in m.modules()), 0)

    def test_svd_center_exact_rank_error_and_full_rank_oracle(self):
        weight, bias = torch.randn(32, 32, 3, 3), torch.randn(32)
        errors = []
        x = torch.randn(2, 32, 11, 14, dtype=torch.float64)
        for ratio in (0.25, 0.5, 1.):
            m = CenterRankConv(32, 32, ratio=ratio).double().eval()
            report = m.initialize_from_kernel(weight, bias)
            errors.append(report['relative_offcenter_error'])
            m.switch_to_deploy()
            spatial = m.spatial.fused_conv.weight
            center, left = m.mix.conv.weight[:, :32, 0, 0], m.mix.conv.weight[:, 32:, 0, 0]
            effective = torch.einsum('or,rihw->oihw', left, spatial)
            effective[:, :, 1, 1] += center
            torch.testing.assert_close(effective[:, :, 1, 1], weight[:, :, 1, 1].double(), rtol=1e-10, atol=1e-10)
            flat = effective.reshape(32, 32, 9)[:, :, [0,1,2,3,5,6,7,8]].reshape(32, -1)
            self.assertLessEqual(int(torch.linalg.matrix_rank(flat, tol=1e-7)), m.rank)
            if ratio == 1.:
                expected = torch.nn.functional.silu(torch.nn.functional.conv2d(x, weight.double(), bias.double(), padding=1))
                torch.testing.assert_close(m(x), expected, rtol=1e-10, atol=1e-10)
                down = CenterRankConv(32, 32, s=2, ratio=1.).double().eval()
                down.initialize_from_kernel(weight, bias)
                down.switch_to_deploy()
                expected = torch.nn.functional.silu(torch.nn.functional.conv2d(x, weight.double(), bias.double(), stride=2, padding=1))
                torch.testing.assert_close(down(x), expected, rtol=1e-10, atol=1e-10)
        self.assertGreater(errors[0], errors[1])
        self.assertLess(errors[2], 1e-10)

    def test_all_ablations_real_loss_and_deploy(self):
        batch = dict(img=torch.randn(2, 3, 64, 96), batch_idx=torch.tensor([0., 1.]),
                     cls=torch.tensor([[0.], [1.]]), bboxes=torch.tensor([[.5, .5, .4, .4], [.3, .4, .25, .3]]))
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                m = build(variant).train()
                m.args = get_cfg()
                loss, _ = m(batch)
                loss.sum().backward()
                self.assertTrue(torch.isfinite(loss).all())
                grads = [p.grad for p in m.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                m.eval()
                with torch.inference_mode():
                    before = m(batch['img'])[1]['one2one']
                    deployed = deepcopy(m).fuse(verbose=False)
                    after = deployed(batch['img'])[1]['one2one']
                for key in ('boxes', 'scores'):
                    torch.testing.assert_close(after[key], before[key], rtol=1e-4, atol=1e-4)
                self.assertFalse(any(hasattr(b, 'deployed') and not b.deployed for b in deployed.modules()))

    def test_svd_model_transfer_unaffected_tensors_and_no_source_mutation(self):
        source = build('baseline').eval()
        original = deepcopy(source.state_dict())
        target = build('abc').eval()
        report = transfer_orthogonal_rank(source, target)
        self.assertFalse(report['exact_checkpoint_transfer'])
        self.assertGreater(len(report['approximated_layers']), 17)
        torch.testing.assert_close(source.state_dict(), original, rtol=0, atol=0)
        for key, value in source.model[-1].cv3.state_dict().items():
            torch.testing.assert_close(value, target.model[-1].cv3.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(2, 3, 64, 96)
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(target(x)[0]).all())
        before = deepcopy(target.state_dict())
        with self.assertRaises(ValueError):
            transfer_orthogonal_rank(build('baseline', nc=80), target)
        torch.testing.assert_close(target.state_dict(), before, rtol=0, atol=0)

    def test_scale_topology_matches_baseline_and_yaml_api_roundtrip(self):
        for scale in 'nsmlx':
            old, new = build('baseline', scale=scale), build('abc', scale=scale)
            self.assertEqual(old.stride.tolist(), new.stride.tolist())
            self.assertIsInstance(new.model[-1], OrthoRankDetect)
            for i in (4, 6, 8, 12, 14, 17, 20):
                self.assertEqual(type(old.model[i].m[0]), type(new.model[i].m[0]))
            del old, new
        model = YOLO(str(CONFIGS/'yolo11-orthogonal-rank-bc.yaml'))
        with tempfile.TemporaryDirectory(dir=ROOT/'experiments') as tmp:
            path = Path(tmp)/'rank.pt'
            model.save(path)
            loaded = YOLO(path).model.eval()
            expected = deepcopy(model.model).half().float().eval()
            x = torch.randn(1, 3, 96, 128)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(x), expected(x), rtol=0, atol=0)

    def test_cpu_amp_and_non_end2end(self):
        m = build('abc').train()
        x = torch.randn(2, 3, 64, 96)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            raw = m(x)
            loss = sum(v[k].float().square().mean() for v in raw.values() for k in ('boxes', 'scores'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad))
        cfg = variant_config('bc', nc=4)
        cfg['end2end'] = False
        m = DetectionModel(cfg, verbose=False).eval()
        with torch.inference_mode():
            y = m(x)[0]
        self.assertEqual(y.shape, (2, 8, 126))
        self.assertTrue(torch.isfinite(y).all())

    def test_baseline_pre_edit_snapshots(self):
        # Construct new model first to exercise parser/global-state isolation.
        build('abc')
        for name in ('exp000_audit/baseline_reference.pt', 'orthogonal_audit/before_changes.pt'):
            ref = torch.load(ROOT/'experiments'/name, map_location='cpu', weights_only=True)
            m = DetectionModel(deepcopy(ref['config']), verbose=False).eval()
            m.load_state_dict(ref['state_dict'], strict=True)
            with torch.inference_mode():
                actual = m(ref['input'])
            if isinstance(ref['output'], torch.Tensor):
                actual = actual[0]
            torch.testing.assert_close(actual, ref['output'], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Installed torch has no CUDA support')
    def test_cuda_amp_backward_and_half_fuse(self):
        m = CenterRankConv(32, 64, s=2).cuda().train()
        x = torch.randn(2, 32, 13, 16, device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            y = m(x)
        y.float().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        m.half().eval()
        with torch.inference_mode():
            before = m(x.half())
            m.switch_to_deploy()
            torch.testing.assert_close(m(x.half()), before, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
