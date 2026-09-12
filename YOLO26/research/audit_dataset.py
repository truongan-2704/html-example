"""Read-only YOLO detection-label and exact image-duplicate audit.

No download, image cache creation, relabeling or split mutation is performed.
Small/medium/large AP cannot be inferred from these label counts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    root = Path(config.get('path', args.data.resolve().parent)).resolve()
    hashes, origins, summary, errors = defaultdict(set), defaultdict(set), {}, []
    for split in ('train', 'val', 'test'):
        relative = config.get(split)
        if not isinstance(relative, str):
            raise ValueError('This audit supports one image directory per split')
        images = (root / relative).resolve()
        if not images.exists() and relative.startswith('../'):
            images = (root / relative[3:]).resolve()
        if not images.is_dir():
            raise FileNotFoundError(images)
        label_dir = images.parent / 'labels'
        counts, negatives, missing, digest, image_count = Counter(), 0, [], hashlib.sha256(), 0
        for image_path in sorted(images.iterdir()):
            if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            image_count += 1
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            hashes[image_hash].add(split)
            origins[image_path.stem.split('.rf.')[0]].add(split)
            label = label_dir / (image_path.stem + '.txt')
            if not label.exists():
                missing.append(image_path.name)
                continue
            raw = label.read_bytes()
            digest.update(image_path.name.encode() + b'\0' + image_hash.encode() + b'\0' + raw)
            lines = raw.decode('utf-8').splitlines()
            if not any(line.strip() for line in lines):
                negatives += 1
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    values = [float(v) for v in line.split()]
                    assert len(values) == 5 and all(math.isfinite(v) for v in values)
                    cls, cx, cy, width, height = values
                    assert cls.is_integer() and 0 <= cls < config['nc']
                    assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1
                    counts[int(cls)] += 1
                except (ValueError, AssertionError):
                    errors.append(dict(file=str(label), line=line_number, text=line))
        summary[split] = dict(images=image_count, directory=str(images), class_instances=dict(counts),
                              empty_labels=negatives, missing_labels=missing, split_manifest_sha256=digest.hexdigest())
    result = dict(data=str(args.data.resolve()), config=config, splits=summary, invalid_labels=errors,
                  cross_split_identical_image_hashes={h:sorted(s) for h,s in hashes.items() if len(s)>1},
                  cross_split_original_filename_candidates={h:sorted(s) for h,s in origins.items() if len(s)>1},
                  limitations='Exact bytes and filename heuristics only. No perceptual dedup, group/scene leakage audit, image decode validation or annotation correctness verification.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(splits=summary, invalid_labels=len(errors),
                         exact_cross_split_duplicates=len(result['cross_split_identical_image_hashes']),
                         filename_candidates=len(result['cross_split_original_filename_candidates'])), indent=2))


if __name__ == '__main__':
    main()
