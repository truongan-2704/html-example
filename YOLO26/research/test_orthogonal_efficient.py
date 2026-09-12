"""Executable unittest suite; no pytest installation or pretrained download required."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.orthogonal_blocks import OHRConv, OSIFusion
from ultralytics.nn.modules.orthogonal_efficient import OSIFusionEfficient
from ultralytics.nn.modules.SimAM import SimAM
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

CONFIG_DIR = ROOT / 'ultralytics/cfg/models/11/yolo11-Orthogonal'


def build(name='yolo11-orthogonal-efficient.yaml', scale='n'):
    cfg = YAML.load(CONFIG_DIR / name)
    cfg['scale'] = scale
    return DetectionModel(cfg, verbose=False)


def random_bn(module):
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            with torch.no_grad():
                m.running_mean.uniform_(-.4, .4)
                m.running_var.uniform_(.4, 1.8)
                m.weight.uniform_(.5, 1.5)
                m.bias.uniform_(-.2, .2)
                m.eps = .002


class OrthogonalEfficientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_ohr_fuse_dtype_groups_stride_bn_and_idempotence(self):
        for dtype in (torch.float32, torch.float64):
            for c1, c2, groups, stride in ((8,8,1,1), (8,16,2,1), (8,8,8,1), (8,16,2,2)):
                with self.subTest(dtype=dtype, groups=groups, stride=stride):
                    m = OHRConv(c1,c2,g=groups,s=stride).to(dtype=dtype).eval()
                    random_bn(m)
                    x = torch.randn(2,c1,9,12,dtype=dtype)
                    with torch.inference_mode():
                        expected = m(x)
                        m.fuse()
                        actual = m(x)
                        m.fuse()
                    self.assertEqual(m.fused_conv.weight.dtype,dtype)
                    self.assertEqual(m.fused_conv.weight.device,x.device)
                    self.assertFalse(m.fused_conv.training)
                    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
                    self.assertTrue(torch.isfinite(actual).all())

    def test_eval_equivalence_shapes_weights_and_no_input_mutation(self):
        for channels in ((8,16,8),(8,8,8),(8,16,12)):
            for fine, coarse in (((9,13),(5,7)),((8,12),(4,6)),((5,7),(5,7)),((5,7),(9,13)),((5,13),(9,7))):
                with self.subTest(channels=channels,fine=fine,coarse=coarse):
                    old = OSIFusion(*channels).double().eval()
                    random_bn(old)
                    old.w.data.copy_(torch.tensor([.2,1.4],dtype=torch.float64))
                    new = OSIFusionEfficient(*channels).double().eval()
                    new.load_state_dict(old.state_dict(),strict=True)
                    # eps is module metadata, not state_dict, so match explicitly.
                    for m in new.modules():
                        if isinstance(m,torch.nn.BatchNorm2d):
                            m.eps=.002
                    xs = [torch.randn(2,channels[0],*fine,dtype=torch.float64),torch.randn(2,channels[1],*coarse,dtype=torch.float64)]
                    copies = [v.clone() for v in xs]
                    with torch.inference_mode():
                        expected, actual = old(xs), new(xs)
                    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
                    for a,b in zip(xs,copies):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_training_outputs_gradients_bn_exact(self):
        old=OSIFusion(8,16,8).train()
        new=OSIFusionEfficient(8,16,8).train()
        new.load_state_dict(old.state_dict(),strict=True)
        x=[torch.randn(2,8,9,13,requires_grad=True),torch.randn(2,16,5,7,requires_grad=True)]
        z=[v.detach().clone().requires_grad_(True) for v in x]
        a,b=old(x),new(z)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        for p,q in zip(old.parameters(),new.parameters()):
            self.assertIsNotNone(q.grad)
            self.assertTrue(torch.isfinite(q.grad).all())
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        for p,q in zip(x,z):
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        for key,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)

    def test_cpu_amp_forward_backward(self):
        m=OSIFusionEfficient(8,16,8).train()
        x=[torch.randn(2,8,9,13),torch.randn(2,16,5,7)]
        with torch.autocast('cpu',dtype=torch.bfloat16):
            y=m(x)
            loss=y.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_projection_runs_at_native_resolution(self):
        m=OSIFusionEfficient(8,16,8).eval()
        seen=[]
        handle=m.proj_glb.register_forward_pre_hook(lambda m,x:seen.append(tuple(x[0].shape[-2:])))
        m([torch.randn(2,8,9,13),torch.randn(2,16,5,7)])
        handle.remove()
        self.assertEqual(seen,[(5,7)])

    def test_singleton_spatial_feature_finite(self):
        x=torch.randn(2,8,1,1,requires_grad=True)
        y=SimAM()(x)
        torch.testing.assert_close(y,x*torch.sigmoid(torch.tensor(.5)))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        model=build().eval()
        with torch.inference_mode():
            output=model(torch.randn(2,3,32,32))
        self.assertTrue(torch.isfinite(output[0]).all())

    def test_model_yaml_transfer_loss_and_fuse(self):
        old=build('yolo11-orthogonal.yaml').eval()
        new=build().eval()
        new.load_state_dict(old.state_dict(),strict=True)
        x=torch.randn(2,3,64,96)
        with torch.inference_mode():
            a,b=old(x),new(x)
        for branch in ('one2one','one2many'):
            for key in ('boxes','scores'):
                torch.testing.assert_close(a[1][branch][key],b[1][branch][key],rtol=1e-5,atol=1e-5)
        new.train()
        new.args=get_cfg()
        batch=dict(img=x,batch_idx=torch.tensor([0.,1.]),cls=torch.tensor([[0.],[1.]]),bboxes=torch.tensor([[.5,.5,.4,.4],[.4,.4,.3,.3]]))
        losses,_=new(batch)
        losses.sum().backward()
        self.assertTrue(torch.isfinite(losses).all())
        grads=[p.grad for p in new.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        new.eval()
        with torch.inference_mode():
            raw=new(x)[1]['one2one']
            fused=deepcopy(new).fuse(verbose=False)
            after=fused(x)[1]['one2one']
        for key in ('boxes','scores'):
            torch.testing.assert_close(raw[key],after[key],rtol=1e-4,atol=1e-4)

    def test_all_yaml_scales_construct(self):
        # Build all scales; avoid huge 640 inputs when validating parser contracts.
        for scale in 'nsmlx':
            with self.subTest(scale=scale):
                model=build(scale=scale)
                self.assertEqual(model.stride.tolist(),[8.,16.,32.])
                self.assertEqual(sum(isinstance(m,OSIFusionEfficient) for m in model.modules()),4)
                del model

    def test_yolo_api_checkpoint_roundtrip(self):
        model=YOLO(str(CONFIG_DIR/'yolo11-orthogonal-efficient.yaml'))
        model.model.eval()
        x=torch.randn(1,3,64,96)
        with torch.inference_mode():
            expected=model.model(x)[1]['one2one']['boxes']
        with tempfile.TemporaryDirectory(dir=ROOT/'experiments') as tmp:
            path=Path(tmp)/'model.pt'
            model.save(path)
            loaded=YOLO(path).model.eval()
            with torch.inference_mode():
                actual=loaded(x)[1]['one2one']['boxes']
        # YOLO.save stores FP16 weights by design; compare with that serialization tolerance.
        torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-3)

    def test_existing_snapshots(self):
        for path in (ROOT/'experiments/exp000_audit/baseline_reference.pt',ROOT/'experiments/orthogonal_audit/before_changes.pt'):
            with self.subTest(path=path.name):
                ref=torch.load(path,map_location='cpu',weights_only=True)
                m=DetectionModel(deepcopy(ref['config']),verbose=False).eval()
                m.load_state_dict(ref['state_dict'],strict=True)
                with torch.inference_mode():
                    actual=m(ref['input'])
                expected=ref['output']
                if isinstance(expected,torch.Tensor):
                    actual=actual[0]
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable in installed PyTorch build')
    def test_cuda_amp_and_fuse(self):
        m=OHRConv(8,8).cuda().half().eval()
        x=torch.randn(2,8,9,13,device='cuda',dtype=torch.float16)
        with torch.inference_mode():
            a=m(x)
            m.fuse()
            torch.testing.assert_close(m(x),a,rtol=1e-2,atol=1e-2)
        fusion=OSIFusionEfficient(8,16,8).cuda().train()
        inputs=[torch.randn(2,8,9,13,device='cuda'),torch.randn(2,16,5,7,device='cuda')]
        with torch.autocast('cuda',dtype=torch.float16):
            y=fusion(inputs)
        y.float().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
