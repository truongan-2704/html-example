"""Paired CPU FP32 benchmark of original and efficient Orthogonal (same random weights).

Model-only latency includes the built-in end-to-end top-k head, but no preprocessing/IO.
No training or accuracy evaluation; results only describe this hardware/run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=60)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--selected-ablation',action='store_true',help='Include selected-decode alone and combined with efficient fusion')
    parser.add_argument('--output',type=Path,help='JSON destination; defaults to the original audit benchmark path')
    args=parser.parse_args()
    if args.samples < 10 or args.imgsz < 32:
        parser.error('Use at least 10 samples and imgsz >= 32')
    import numpy as np
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import YAML

    torch.set_num_threads(1)
    torch.manual_seed(7)
    rng=random.Random(7)
    out=ROOT/'experiments/orthogonal_audit'
    out.mkdir(parents=True,exist_ok=True)
    ref=torch.load(out/'before_changes.pt',map_location='cpu',weights_only=True)
    old=DetectionModel(deepcopy(ref['config']),verbose=False).eval()
    new_cfg=YAML.load(ROOT/'ultralytics/cfg/models/11/yolo11-Orthogonal/yolo11-orthogonal-efficient.yaml')
    new_cfg['scale']='n'
    new=DetectionModel(new_cfg,verbose=False).eval()
    models={'original':old,'efficient':new}
    if args.selected_ablation:
        for name,filename in (('selected','yolo11-orthogonal-sd.yaml'),('efficient_selected','yolo11-orthogonal-efficient-sd.yaml')):
            cfg=YAML.load(ROOT/'ultralytics/cfg/models/11/yolo11-Orthogonal'/filename)
            cfg['scale']='n'
            models[name]=DetectionModel(cfg,verbose=False).eval()
    for model in models.values():
        model.load_state_dict(ref['state_dict'],strict=True)
        model.fuse(verbose=False)
    x=torch.randn(1,3,args.imgsz,args.imgsz)
    results={}
    for name,model in models.items():
        operations=[]
        dfl_inputs=[]
        def count(m,inputs,output):
            operations.append(2*output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1])
        handles=[m.register_forward_hook(count) for m in model.modules() if isinstance(m,torch.nn.Conv2d)]
        handles.append(model.model[-1].dfl.register_forward_pre_hook(lambda m,inputs:dfl_inputs.append(list(inputs[0].shape))))
        with torch.inference_mode():
            prediction=model(x)[0]
        if name=='original':
            reference_prediction=prediction
        torch.testing.assert_close(prediction,reference_prediction,rtol=1e-4,atol=1e-4)
        for h in handles:
            h.remove()
        results[name]=dict(parameters=sum(p.numel() for p in model.parameters()),conv_gflops=sum(operations)/1e9,
                           dfl_input_shapes=dfl_inputs,output_matches_reference=True,samples_ms=[])
    with torch.inference_mode():
        for _ in range(20):
            for m in models.values():
                m(x)
        for _ in range(args.samples):
            order=list(models)
            rng.shuffle(order)
            for name in order:
                start=time.perf_counter_ns()
                models[name](x)
                results[name]['samples_ms'].append((time.perf_counter_ns()-start)/1e6)
    for result in results.values():
        times=np.array(result['samples_ms'])
        result.update(mean_ms=float(times.mean()),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),fps=float(1000/times.mean()))
    # Paired bootstrap describes within-run timing uncertainty, not training variability.
    differences=np.array(results['original']['samples_ms'])-np.array(results['efficient']['samples_ms'])
    brng=np.random.default_rng(7)
    boot=np.mean(brng.choice(differences,(2000,len(differences)),replace=True),axis=1)
    report=dict(environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),processor=platform.processor(),threads=1),
                input=list(x.shape),precision='float32',trained=False,fused=True,warmup_per_model=20,results=results,
                paired_original_minus_efficient_mean_ms=float(differences.mean()),paired_bootstrap_95_interval_ms=np.percentile(boot,[2.5,97.5]).tolist(),
                accuracy='NOT_EVALUATED',scope='CPU eager model-only including top-k; 2 FLOPs/MAC Conv2d only, no SimAM/BN/SiLU/resize/elementwise/decoding/top-k arithmetic in conv count. Randomized alternating pair order. Not GPU/end-to-end or publication evidence.')
    comparisons={}
    for reference,candidate in [('original',name) for name in models if name!='original'] + ([('efficient','efficient_selected')] if args.selected_ablation else []):
        delta=np.array(results[reference]['samples_ms'])-np.array(results[candidate]['samples_ms'])
        boot=np.mean(brng.choice(delta,(2000,len(delta)),replace=True),axis=1)
        comparisons[reference+'_minus_'+candidate]=dict(mean_ms=float(delta.mean()),bootstrap_95_interval_ms=np.percentile(boot,[2.5,97.5]).tolist())
    report['paired_comparisons']=comparisons
    destination=args.output or out/('selected_ablation.json' if args.selected_ablation else 'benchmark.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    print(json.dumps({k:{q:v for q,v in r.items() if q!='samples_ms'} for k,r in results.items()},indent=2))


if __name__=='__main__':
    main()
