"""Train actual Orthogonal-Rank architectures with an explicit reproducible recipe.

Default: capacity-first Orthogonal with full spatial operators. Old rank ablations remain explicit options.
Example: .venv/Scripts/python.exe train_orthogonal_rank.py --variant capacity --device 0 --amp --seeds 0 1 2
Use --prepare-only to write the exact recipe without starting training.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
settings = ROOT / 'experiments/orthogonal_rank/settings'
settings.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(settings))

from orthogonal_rank_common import training_config, TRAIN_VARIANTS, save_source_snapshot
from ultralytics.utils import YAML


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=TRAIN_VARIANTS, default='capacity')
    parser.add_argument('--ratio', type=float, help='Spatial rank ratio for old A/B/C ablations only (default .25)')
    parser.add_argument('--scale', choices=list('nsmlx'), default='n')
    parser.add_argument('--data', type=Path, default=ROOT/'ultralytics/data/apple_leaft_detection/data.yaml')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-plots', action='store_true', help='Disable training plots; useful for minimal smoke-test environments')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--project', type=Path)
    parser.add_argument('--init', type=Path, help='Matching original Orthogonal checkpoint; approximate SVD initialization, not resume')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.variant in ('capacity', 'yolo11') and args.ratio is not None:
        parser.error('--ratio applies to the rank ablations; Capacity and stock YOLO11 have no rank restriction')
    ratio = args.ratio if args.ratio is not None else .25
    if args.project is None:
        family = 'orthogonal_capacity' if args.variant in ('capacity','yolo11') else 'orthogonal_rank'
        args.project = ROOT/'experiments'/family/'training'
    if not args.data.is_file() or args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        parser.error('Provide an existing dataset YAML and positive training dimensions/epochs')
    if args.init and not args.init.is_file():
        parser.error('--init must be an existing local checkpoint')
    data = YAML.load(args.data)
    nc = data.get('nc') or len(data['names'])
    cfg = training_config(args.variant, ratio, args.scale, nc)
    prefix = f'{args.variant}_{args.scale}' if args.variant in ('capacity', 'yolo11') else f'{args.variant}_{args.scale}_r{ratio:g}'
    prepared = args.project/'prepared'/prefix
    prepared.mkdir(parents=True, exist_ok=True)
    # Local yaml_model_load derives scale from filename and overrides YAML['scale'].
    config_path = prepared/f'yolo11{args.scale}-{args.variant}.yaml'
    YAML.save(config_path, cfg)
    recipes = []
    for seed in args.seeds:
        recipes.append(dict(model=str(config_path.resolve()), data=str(args.data.resolve()),
                            imgsz=args.imgsz, batch=args.batch, epochs=args.epochs, cache=False,
                            amp=args.amp, optimizer='SGD', patience=30, save_period=10, seed=seed,
                            project=str(args.project.resolve()), name=f'{prefix}_seed{seed}', workers=args.workers,
                            device=args.device, val=True, deterministic=True, pretrained=False, plots=not args.no_plots))
    YAML.save(prepared/'recipes.yaml', dict(initialization=str(args.init.resolve()) if args.init else 'scratch', recipes=recipes))
    print(f'Prepared model and matched baseline/candidate recipe: {prepared}')
    if args.prepare_only:
        print('No training or mAP evaluation was run.')
        return
    import torch
    if args.device != 'cpu' and not torch.cuda.is_available():
        parser.error('Requested CUDA training, but this PyTorch installation has no CUDA support')
    if not args.no_plots and importlib.util.find_spec('polars') is None:
        parser.error('Training plots require polars in this checkout; install it or use --no-plots (CSV/metrics remain enabled)')
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.modules.orthogonal_rank import transfer_orthogonal_rank

    class RankTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            target = super().get_model(cfg, weights, verbose)
            if args.init:
                source = YOLO(args.init).model.eval()
                if args.variant in ('baseline', 'yolo11'):
                    target.load_state_dict(source.state_dict(), strict=True)
                    report = dict(exact_checkpoint_transfer=True)
                elif args.variant == 'capacity':
                    if len(source.model) != len(target.model) or any(type(a) is not type(b) for a,b in zip(source.model,target.model)):
                        raise ValueError('Capacity initialization requires a matching original Orthogonal architecture')
                    if source.model[-1].nc != target.model[-1].nc or source.model[0].conv.weight.shape != target.model[0].conv.weight.shape:
                        raise ValueError('Capacity initialization requires matching classes and model scale')
                    if source.names != self.data['names']:
                        raise ValueError('Capacity initialization requires matching class names/order; do not reuse an unrelated classification head')
                    src, dst = source.state_dict(), target.state_dict()
                    matched = {k:v for k,v in src.items() if k in dst and v.shape == dst[k].shape}
                    target.load_state_dict(matched, strict=False)
                    report = dict(exact_checkpoint_transfer=len(matched)==len(dst), method='shape-matched initialization; unmatched wider layers need training',
                                  copied_tensors=len(matched), randomly_initialized_tensors=[k for k in dst if k not in matched])
                else:
                    report = transfer_orthogonal_rank(source, target)
                (self.save_dir/'initialization.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            return target

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    for recipe in recipes:
        trainer = RankTrainer(overrides=recipe)
        YAML.save(trainer.save_dir/'architecture.yaml', cfg)
        save_source_snapshot(trainer.save_dir)
        (trainer.save_dir/'command.txt').write_text(subprocess.list2cmdline([sys.executable, *sys.argv])+'\n', encoding='utf-8')
        (trainer.save_dir/'git_commit.txt').write_text(commit+'\nWorking tree source hashes are required alongside this commit.\n', encoding='utf-8')
        (trainer.save_dir/'environment.txt').write_text(f'{sys.version}\n{platform.platform()}\ntorch={torch.__version__}\nCUDA={torch.version.cuda}\nseed={recipe["seed"]}\n',encoding='utf-8')
        packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
        (trainer.save_dir/'packages.txt').write_text(packages, encoding='utf-8')
        trainer.train()
        (trainer.save_dir/'metrics.json').write_text(json.dumps(trainer.metrics, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
