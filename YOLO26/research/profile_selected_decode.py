"""Isolate decode/top-k cost with identical synthetic logits; no detector accuracy claim."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from ultralytics.nn.modules.selected_decode import SelectDecodeDetect

    torch.set_num_threads(1)
    torch.manual_seed(23)
    rng = random.Random(23)
    head = SelectDecodeDetect(80, 16, True, (8, 16, 32)).eval()
    head.stride = torch.tensor([8., 16., 32.])
    preds = dict(feats=[torch.randn(1, c, h, h) for c, h in ((8, 80), (16, 40), (32, 20))],
                 boxes=torch.randn(1, 64, 8400) * 2, scores=torch.randn(1, 80, 8400))
    methods = dict(decode_all=lambda: head.postprocess(head._inference(preds).permute(0, 2, 1)),
                   select_first=lambda: head._selected_inference(preds))
    times = {name: [] for name in methods}
    with torch.inference_mode():
        torch.testing.assert_close(methods['decode_all'](), methods['select_first'](), rtol=1e-5, atol=1e-4)
        for _ in range(50):
            for fn in methods.values():
                fn()
        for _ in range(200):
            order = list(methods)
            rng.shuffle(order)
            for name in order:
                start = time.perf_counter_ns()
                methods[name]()
                times[name].append((time.perf_counter_ns() - start) / 1e6)
    result = dict(scope='Only decode + top-k from synthetic precomputed logits, cached anchors; excludes regression/class convolutions and all other model stages.',
                  torch=torch.__version__,threads=1,device='cpu',precision='float32',batch=1,nc=80,reg_max=16,positions=8400,max_det=300,
                  warmup=50,samples=200,trained=False,results={})
    for name, values in times.items():
        a = np.array(values)
        result['results'][name] = dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                                       p95_ms=float(np.percentile(a, 95)), samples_ms=values)
    differences = np.array(times['decode_all']) - np.array(times['select_first'])
    boot = np.random.default_rng(23).choice(differences,(2000,len(differences)),replace=True).mean(1)
    result['paired_saving_ms'] = float(differences.mean())
    result['within_run_bootstrap_95_interval_ms'] = np.percentile(boot,[2.5,97.5]).tolist()
    result['derived_dfl_probability_tensor_bytes'] = dict(decode_all=1*64*8400*4,select_first=1*64*300*4,
        note='Calculated tensor size, NOT measured peak memory. Raw dense logits and feature maps are still allocated.')
    out = ROOT / 'experiments/orthogonal_selected'
    out.mkdir(parents=True,exist_ok=True)
    (out/'decode_microbenchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k: v if k != 'results' else {n: {q:r for q,r in d.items() if q != 'samples_ms'} for n,d in v.items()} for k,v in result.items()},indent=2))


if __name__ == '__main__':
    main()
