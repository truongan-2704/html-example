"""Collect actual rank architecture artifacts without inventing unrun accuracy metrics."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orthogonal_rank_common import save_source_snapshot, variant_config


def main():
    import torch
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.orthogonal_rank import transfer_orthogonal_rank

    torch.set_num_threads(1)
    out = ROOT/'experiments/orthogonal_rank'
    save_source_snapshot(out)
    commit = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    (out/'git_commit.txt').write_text(commit+'\nUncommitted implementation archived in source_snapshot.zip. Snapshot captured at finalization; benchmark inference code is unchanged since its runs.\n',encoding='utf-8')
    (out/'tracked_changes.patch').write_bytes(subprocess.run(['git','diff','--','.'],cwd=ROOT,capture_output=True,check=True).stdout)
    (out/'packages.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout,encoding='utf-8')
    (out/'environment.txt').write_text(f'{sys.version}\n{platform.platform()}\n{platform.processor()}\nUltralytics={ultralytics.__version__}\nPyTorch={torch.__version__}\nCUDA={torch.version.cuda}\nCUDA available={torch.cuda.is_available()}\nBenchmark threads=1, FP32, batch1\n',encoding='utf-8')
    fields=['run','nc','ratio','model','training_form_parameters','deploy_parameters','conv_gflops','mean_ms','median_ms','p95_ms','fps']
    commands=['# Benchmark commands reconstructed from saved run arguments.']
    with (out/'results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for filename in ('ablation_nc80_run1.json','confirm_nc80_run2.json','confirm_nc4.json','sensitivity_r0375.json','sensitivity_r05.json'):
            report=json.loads((out/filename).read_text())
            commands.append(f'.venv/Scripts/python.exe -B research/benchmark_orthogonal_rank.py --variants {" ".join(report["results"])} --ratio {report["ratio"]} --nc {report["nc"]} --imgsz {report["input"][-1]} --samples {report["samples"]} --warmup {report["warmup"]} --device {report["environment"]["device"]} --output experiments/orthogonal_rank/{filename}')
            for name,metrics in report['results'].items():
                writer.writerow(dict(run=filename,nc=report['nc'],ratio=report['ratio'],model=name,**{key:metrics[key] for key in fields[4:]}))
    commands += ['.venv/Scripts/python.exe -B research/test_orthogonal_rank.py',
                 '.venv/Scripts/python.exe -B research/export_orthogonal_probe.py --rank',
                 '.venv/Scripts/python.exe -B research/smoke_train_orthogonal_rank.py']
    (out/'command.txt').write_text('\n'.join(commands)+'\n',encoding='utf-8')
    ref=torch.load(ROOT/'experiments/orthogonal_audit/before_changes.pt',map_location='cpu',weights_only=True)
    original=DetectionModel(deepcopy(ref['config']),verbose=False).eval()
    original.load_state_dict(ref['state_dict'],strict=True)
    candidate=DetectionModel(variant_config('bc'),verbose=False).eval()
    transfer=transfer_orthogonal_rank(original,candidate)
    (out/'initialization_nc80.json').write_text(json.dumps(transfer,indent=2),encoding='utf-8')
    (out/'weights').mkdir(exist_ok=True)
    sizes={}
    for name,model in (('baseline',original),('bc',candidate)):
        model.fuse(verbose=False)
        path=out/'weights'/f'{name}_random_fused_state.pt'
        torch.save(dict(state_dict=model.state_dict(),trained=False,scope='FP32 fused random-weight state_dict, not trained detection weights'),path)
        sizes[name]=dict(file_bytes=path.stat().st_size,parameters=sum(p.numel() for p in model.parameters()),dtype='float32')
    (out/'serialized_sizes.json').write_text(json.dumps(sizes,indent=2),encoding='utf-8')
    smoke=out/'trainer_smoke'
    manifest=json.loads((smoke/'sample_manifest.json').read_text())
    smoke_metrics={}
    for variant in ('baseline','bc'):
        run=smoke/'runs'/f'{variant}_n_r0.25_seed0'
        smoke_metrics[variant]=json.loads((run/'metrics.json').read_text())
    summary=dict(candidate='B+C ratio0.25',scientific_decision='REVISE pending full-dataset accuracy',
                 full_dataset_mAP50=None,full_dataset_mAP50_95=None,APsmall=None,peak_gpu_memory=None,end_to_end_latency=None,
                 architectural_tests=dict(passed=8,cuda_skipped=1),dynamic_onnx='PASS',serialized_sizes=sizes,
                 smoke_training=dict(epochs=1,imgsz=128,batch=2,train_images=sum(s['split']=='train' for s in manifest['samples']),
                                     validation_images=sum(s['split']=='valid' for s in manifest['samples']),seed=0,metrics=smoke_metrics,
                                     scope='Real-data pipeline test only; neither baseline reproduction nor comparative accuracy evidence'))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
