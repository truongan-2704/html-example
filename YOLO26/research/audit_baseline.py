"""Read-only source inventory and reproducible YOLO11 baseline probe (no training).

Run from repository root with .venv/Scripts/python.exe -B research/audit_baseline.py.
Artifacts are written only to the requested output directory. No weights download.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/exp000_audit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    # Enumerate all files, but parse only source/config/documentation, not data or binaries.
    counts = {}
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in {'.venv', '.git', '__pycache__', '.idea', 'experiments'})
        for name in sorted(files):
            path = Path(directory) / name
            counts[path.suffix.lower()] = counts.get(path.suffix.lower(), 0) + 1
            if path.suffix not in {'.py', '.yaml', '.yml', '.md'}:
                continue
            data = path.read_bytes()
            item = dict(path=path.relative_to(ROOT).as_posix(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            if path.suffix == '.py':
                try:
                    tree = ast.parse(data.decode('utf-8-sig'))
                    item['symbols'] = [dict(name=n.name, line=n.lineno, kind=type(n).__name__) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
                except (SyntaxError, UnicodeError) as error:
                    item['parse_error'] = str(error)
            inventory.append(item)
    (args.output / 'source_inventory.json').write_text(json.dumps(dict(file_counts=counts, files=inventory), indent=2), encoding='utf-8')

    import torch
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import YAML

    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = YAML.load(ROOT / 'ultralytics/cfg/models/11/yolo11.yaml')
    config['scale'] = 'n'
    model = DetectionModel(config, verbose=False).eval()
    x = torch.randn(1, 3, 640, 640)

    def shape(value):
        if isinstance(value, torch.Tensor):
            return list(value.shape)
        if isinstance(value, dict):
            return {k: shape(v) for k, v in value.items()}
        return [shape(v) for v in value]

    rows, handles = [], []
    for index, module in enumerate(model.model):
        def hook(m, inputs, output, index=index):
            rows.append(dict(index=index, module=type(m).__name__, source=m.f,
                             parameters=sum(p.numel() for p in m.parameters()), output=shape(output)))
        handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        prediction = model(x)[0]
    for handle in handles:
        handle.remove()
    assert torch.isfinite(prediction).all()
    # Snapshot enables exact before/after regression on this environment and input.
    torch.save(dict(state_dict=model.state_dict(), input=x, output=prediction, config=config), args.output / 'baseline_reference.pt')
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        samples = []
        for _ in range(30):
            start = time.perf_counter_ns()
            model(x)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    env = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
               torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
               ultralytics=ultralytics.__version__, import_path=ultralytics.__file__, threads=torch.get_num_threads(),
               git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    result = dict(environment=env, config=config, seed=0, input_shape=list(x.shape), precision='float32',
                  device='cpu', fused=False, trained=False, stages=rows, parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  latency_ms=samples, warmup=5, output_shape=list(prediction.shape),
                  accuracy_status='NOT_EVALUATED', note='Synthetic input, untrained model, CPU model-only wall time; not GPU or end-to-end latency.')
    (args.output / 'baseline_probe.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (args.output / 'environment.txt').write_text(json.dumps(env, indent=2), encoding='utf-8')
    (args.output / 'git_commit.txt').write_text(env['git_commit'] + '\n', encoding='utf-8')
    (args.output / 'command.txt').write_text(subprocess.list2cmdline(sys.argv) + '\n', encoding='utf-8')
    print(json.dumps(dict(parameters=result['parameters'], output_shape=result['output_shape'], environment=env, inventoried_sources=len(inventory)), indent=2))


if __name__ == '__main__':
    main()
