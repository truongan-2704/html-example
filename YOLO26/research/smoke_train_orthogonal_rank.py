"""Run one real-data epoch for baseline and rank B+C; pipeline validation only, not AP evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from ultralytics.utils import YAML
    source = ROOT/'ultralytics/data/apple_leaft_detection'
    out = ROOT/'experiments/orthogonal_rank/trainer_smoke'
    data = out/'data'
    manifest = []
    for split, per_class in (('train', 4), ('valid', 2)):
        selected = {}
        counts = {i: 0 for i in range(4)}
        images = {p.stem: p for p in (source/split/'images').iterdir() if p.is_file()}
        for label in sorted((source/split/'labels').glob('*.txt')):
            classes = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
            if any(counts[c] < per_class for c in classes) and label.stem in images:
                selected[label.stem] = (images[label.stem], label)
                for c in classes:
                    counts[c] += 1
            if all(n >= per_class for n in counts.values()):
                break
        if not all(n >= per_class for n in counts.values()):
            raise RuntimeError(f'Insufficient class coverage in {split}')
        for folder in ('images', 'labels'):
            (data/split/folder).mkdir(parents=True, exist_ok=True)
        for image, label in selected.values():
            shutil.copy2(image, data/split/'images'/image.name)
            shutil.copy2(label, data/split/'labels'/label.name)
            manifest.append(dict(split=split,image=str(image.relative_to(ROOT)),label=str(label.relative_to(ROOT)),
                                 image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
    dataset = YAML.load(source/'data.yaml')
    cfg = dict(path=str(data.resolve()),train='train/images',val='valid/images',nc=dataset['nc'],names=dataset['names'])
    YAML.save(data/'data.yaml',cfg)
    (out/'sample_manifest.json').write_text(json.dumps(dict(scope='Tiny pipeline smoke test, not representative accuracy evaluation',samples=manifest),indent=2),encoding='utf-8')
    commands = []
    for variant in ('baseline','bc'):
        command=[sys.executable,'-B',str(ROOT/'train_orthogonal_rank.py'),'--variant',variant,'--data',str(data/'data.yaml'),
                 '--epochs','1','--batch','2','--imgsz','128','--workers','0','--device','cpu','--seeds','0','--no-plots','--project',str(out/'runs')]
        commands.append(command)
        (out/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
        with (out/f'{variant}_training_log.txt').open('wb') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        print(variant, 'trainer exit code',result.returncode,flush=True)
        if result.returncode:
            print((out/f'{variant}_training_log.txt').read_text(encoding='utf-8',errors='replace')[-8000:])
            raise SystemExit(result.returncode)
    print('Both real-data one-epoch pipelines completed. Do not interpret these subset metrics as accuracy evidence.')


if __name__=='__main__':
    main()
