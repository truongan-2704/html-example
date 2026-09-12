"""Controlled A/B/C architecture generation from the unchanged original Orthogonal YAML."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML

VARIANTS = ('baseline', 'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc')
TRAIN_VARIANTS = (*VARIANTS, 'capacity', 'yolo11')
CONFIGS = ROOT / 'configs/research/orthogonal_rank'


def save_source_snapshot(destination: Path):
    """Archive code/configs with hashes; Git HEAD alone omits uncommitted research files."""
    files = list((ROOT/'ultralytics').rglob('*.py')) + list((ROOT/'ultralytics/cfg').rglob('*.yaml'))
    files += list((ROOT/'research').glob('*.py'))
    files += list(ROOT.glob('train_orthogonal*.py')) + [ROOT/'requirements.txt']
    files += list(CONFIGS.glob('*.yaml'))
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(destination/'source_snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            name = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    (destination/'source_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')


def variant_config(variant='bc', ratio=0.25, scale='n', nc=80):
    """A=CSP spatial mixers, B=box towers, C=downsampling; preserve other config choices."""
    if variant not in VARIANTS or scale not in 'nsmlx' or not 0 < ratio <= 1:
        raise ValueError('Unknown ablation/scale or rank ratio outside (0,1]')
    cfg = YAML.load(ROOT / 'ultralytics/cfg/models/11/yolo11-Orthogonal/yolo11-orthogonal.yaml')
    cfg['scale'], cfg['nc'] = scale, nc
    nodes = cfg['backbone'] + cfg['head']
    switches = '' if variant == 'baseline' else variant
    if 'a' in switches:
        for row in nodes:
            if row[2] == 'C3k2_Ortho':
                channels, c3k = row[3][:2]
                expansion = row[3][2] if len(row[3]) > 2 else 0.5
                row[2], row[3] = 'C3k2_OrthoRank', [channels, c3k, expansion, 1, True, ratio]
    if 'b' in switches:
        if nodes[-1][2] != 'Detect':
            raise ValueError('Original config no longer has the expected Detect head')
        nodes[-1][2], nodes[-1][3] = 'OrthoRankDetect', ['nc', ratio]
    if 'c' in switches:
        for i in (3, 5, 7, 15, 18):
            if nodes[i][2] != 'Conv' or nodes[i][3][1:] != [3, 2]:
                raise ValueError(f'Original downsampler {i} changed; repeat the architecture audit')
            nodes[i][2] = 'CenterRankConv'
            nodes[i][3].append(ratio)
    return cfg


def capacity_config(scale='n', nc=80):
    """Restore full spatial operators and spend capacity on existing P3/P4/P5 refinement.

    This is a separate configuration, not a claim of increased accuracy. At n scale
    its measured parameter count remains below stock YOLO11n at nc4 and nc80.
    Other scales require their own budget checks.
    """
    cfg = variant_config('baseline', scale=scale, nc=nc)
    nodes = cfg['backbone'] + cfg['head']
    nodes[8][3] = [1024, False, .375]  # P5 backbone: .25 -> .375
    nodes[14][3] = [256, False, .5]    # P3 neck: .375 -> .5
    for i in (12, 17):
        nodes[i][3] = [512, False, .375]  # P4 refinement: .25 -> .375
    return cfg


def training_config(variant='capacity', ratio=.25, scale='n', nc=80):
    """Keep stock YOLO11, original Orthogonal and candidate recipes explicitly distinct."""
    if variant == 'capacity':
        return capacity_config(scale, nc)
    if variant == 'yolo11':
        cfg = YAML.load(ROOT/'ultralytics/cfg/models/11/yolo11.yaml')
        cfg['scale'], cfg['nc'] = scale, nc
        return cfg
    return variant_config(variant, ratio, scale, nc)


if __name__ == '__main__':
    for variant in VARIANTS:
        YAML.save(CONFIGS / f'yolo11-orthogonal-rank-{variant}.yaml', variant_config(variant))
    for ratio in (0.375, 0.5):
        YAML.save(CONFIGS / f'yolo11-orthogonal-rank-bc-r{str(ratio).replace(".", "")}.yaml', variant_config('bc', ratio))
    print(f'Wrote 8 ablations and 2 rank sensitivity configurations to {CONFIGS}')
