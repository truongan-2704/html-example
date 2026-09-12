"""Measure architectural A/B/C ablations with a fixed input and matched source initialization."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orthogonal_rank_common import variant_config, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--ratio', type=float, default=.25)
    parser.add_argument('--nc', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 10 or args.warmup < 1 or args.imgsz < 32 or args.imgsz % 32 or args.nc < 1:
        parser.error('Need samples>=10, warmup>=1, positive nc and imgsz divisible by32')
    import numpy as np
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.orthogonal_rank import transfer_orthogonal_rank

    torch.set_num_threads(1)
    torch.manual_seed(7)
    rng = random.Random(7)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA not available in this PyTorch build')
    ref = torch.load(ROOT/'experiments/orthogonal_audit/before_changes.pt',map_location='cpu',weights_only=True)
    source = DetectionModel(variant_config('baseline',nc=args.nc),verbose=False).eval()
    if args.nc == 80:
        source.load_state_dict(ref['state_dict'],strict=True)
    models, results, transfers = {}, {}, {}
    for name in dict.fromkeys(['baseline'] + args.variants):
        if name == 'baseline':
            m = deepcopy(source)
        else:
            m = DetectionModel(variant_config(name, args.ratio, nc=args.nc), verbose=False).eval()
            transfers[name] = transfer_orthogonal_rank(source, m)
        training_parameters = sum(p.numel() for p in m.parameters())
        m.fuse(verbose=False).to(device)
        models[name] = m
        results[name] = dict(training_form_parameters=training_parameters, deploy_parameters=sum(p.numel() for p in m.parameters()),samples_ms=[])
    x = torch.randn(1,3,args.imgsz,args.imgsz,device=device)
    with torch.inference_mode():
        for name,m in models.items():
            operations=[]
            def count(module,inputs,output):
                operations.append(2*output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1])
            hooks=[v.register_forward_hook(count) for v in m.modules() if isinstance(v,torch.nn.Conv2d)]
            y=m(x)[0]
            for hook in hooks:
                hook.remove()
            assert y.shape==(1,min(300,21*(args.imgsz//32)**2),6) and torch.isfinite(y).all()
            results[name]['conv_gflops']=sum(operations)/1e9
            results[name]['output_shape']=list(y.shape)
            results[name]['peak_cuda_allocated_bytes']=None
        for _ in range(args.warmup):
            for m in models.values():
                m(x)
        for _ in range(args.samples):
            order=list(models)
            rng.shuffle(order)
            for name in order:
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                start=time.perf_counter_ns()
                models[name](x)
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                results[name]['samples_ms'].append((time.perf_counter_ns()-start)/1e6)
        if device.type=='cuda':
            # Incremental allocations over resident models/input, not whole-process memory.
            for name,m in models.items():
                torch.cuda.synchronize(device)
                resident=torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                m(x)
                torch.cuda.synchronize(device)
                results[name]['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(device)-resident
    comparisons={}
    for name,result in results.items():
        times=np.array(result['samples_ms'])
        result.update(mean_ms=float(times.mean()),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),fps=float(1000/times.mean()))
        if name!='baseline':
            delta=np.array(results['baseline']['samples_ms'])-times
            boot=np.random.default_rng(7).choice(delta,(2000,len(delta)),replace=True).mean(1)
            comparisons[name]=dict(mean_saving_ms=float(delta.mean()),within_run_bootstrap95_ms=np.percentile(boot,[2.5,97.5]).tolist())
    report=dict(environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),processor=platform.processor(),threads=1,device=str(device)),
                input=list(x.shape),nc=args.nc,ratio=args.ratio,precision='float32',seed=7,trained=False,warmup=args.warmup,samples=args.samples,
                source_initialization='preserved random snapshot' if args.nc==80 else 'seed7 random initialization at requested nc',
                results=results,paired_baseline_minus_candidate=comparisons,approximate_weight_transfers=transfers,
                scope='Fused eager model-only with top-k; randomized paired order. Conv2d only at 2 FLOPs/MAC, not all arithmetic. Excludes preprocessing/IO. No accuracy comparison; architectures have different functions. CUDA memory field, when available, is incremental allocation above all resident models/input, not total model memory.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in row.items() if k!='samples_ms'} for name,row in results.items()},indent=2))


if __name__=='__main__':
    main()
