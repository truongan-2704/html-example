"""Verify capacity-first config, budget versus stock YOLO11 and matched raw-head CPU latency."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from orthogonal_rank_common import training_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nc',type=int,default=80)
    parser.add_argument('--samples',type=int,default=40)
    args=parser.parse_args()
    import numpy as np
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.orthogonal_rank import CenterRankConv

    torch.set_num_threads(1)
    rng=random.Random(17)
    out=ROOT/'experiments/orthogonal_capacity'
    out.mkdir(parents=True,exist_ok=True)
    models,results={},{}
    for name in ('yolo11','baseline','bc','capacity'):
        torch.manual_seed(17)
        m=DetectionModel(training_config(name,nc=args.nc),verbose=False)
        results[name]=dict(training_parameters=sum(p.numel() for p in m.parameters()),samples_ms=[])
        if name=='capacity':
            assert not any(isinstance(b,CenterRankConv) for b in m.modules())
            assert type(m.model[-1]).__name__=='Detect'
            m.args=get_cfg()
            x=torch.randn(2,3,64,96)
            batch=dict(img=x,batch_idx=torch.tensor([0.,1.]),cls=torch.tensor([[0.],[1.]]),
                       bboxes=torch.tensor([[.5,.5,.4,.4],[.4,.4,.3,.3]]))
            loss,_=m(batch)
            loss.sum().backward()
            assert torch.isfinite(loss).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
            m.zero_grad(set_to_none=True)
            m.eval()
            with torch.inference_mode():
                before=m(x)[1]['one2one']
                deployed=deepcopy(m).fuse(verbose=False)
                after=deployed(x)[1]['one2one']
            for key in ('boxes','scores'):
                torch.testing.assert_close(after[key],before[key],rtol=1e-4,atol=1e-4)
            results[name]['real_loss_backward_fuse_checks']='PASS'
        m.eval().fuse(verbose=False)
        results[name]['deploy_parameters']=sum(p.numel() for p in m.parameters())
        # Common comparison endpoint: dense box/class logits. Detect.forward's
        # training return avoids decode/top-k/NMS, while all BN children stay eval.
        m.model[-1].training=True
        assert all(not b.training for b in m.modules() if isinstance(b,torch.nn.BatchNorm2d))
        models[name]=m
    assert results['capacity']['training_parameters']<results['yolo11']['training_parameters']
    assert results['capacity']['deploy_parameters']<results['yolo11']['deploy_parameters']
    x=torch.randn(1,3,640,640)
    with torch.inference_mode():
        for name,m in models.items():
            counts=[]
            def count(mod,inputs,y):
                counts.append(2*y.numel()*(mod.in_channels//mod.groups)*mod.kernel_size[0]*mod.kernel_size[1])
            hooks=[v.register_forward_hook(count) for v in m.modules() if isinstance(v,torch.nn.Conv2d)]
            raw=m(x)
            for h in hooks:h.remove()
            if name!='yolo11':raw=raw['one2one']
            assert raw['boxes'].shape==(1,64,8400) and raw['scores'].shape==(1,args.nc,8400)
            assert torch.isfinite(raw['boxes']).all() and torch.isfinite(raw['scores']).all()
            results[name]['raw_conv_gflops']=sum(counts)/1e9
        for _ in range(15):
            for m in models.values():m(x)
        for _ in range(args.samples):
            order=list(models);rng.shuffle(order)
            for name in order:
                start=time.perf_counter_ns();models[name](x)
                results[name]['samples_ms'].append((time.perf_counter_ns()-start)/1e6)
    for name,r in results.items():
        t=np.array(r['samples_ms'])
        r.update(mean_ms=float(t.mean()),median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),fps=float(1000/t.mean()))
    delta=np.array(results['yolo11']['samples_ms'])-np.array(results['capacity']['samples_ms'])
    boot=np.random.default_rng(17).choice(delta,(2000,len(delta)),replace=True).mean(1)
    report=dict(torch=torch.__version__,device='cpu',threads=1,nc=args.nc,input=list(x.shape),precision='float32',warmup=15,samples=args.samples,trained=False,
                scope='Fused backbone + neck + dense raw head only. No DFL/decode/top-k/NMS, preprocessing or IO for any model. Stock one-to-many versus Orthogonal fused one-to-one; assignment and accuracy still differ. Conv-only arithmetic excludes stock attention matmuls and elementwise work.',
                paired_stock_minus_capacity_ms=float(delta.mean()),within_run_bootstrap95_ms=np.percentile(boot,[2.5,97.5]).tolist(),results=results)
    (out/f'validation_nc{args.nc}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in r.items() if k!='samples_ms'} for name,r in results.items()},indent=2))


if __name__=='__main__':main()
