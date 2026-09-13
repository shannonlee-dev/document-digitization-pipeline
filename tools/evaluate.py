"""수동 평가용 처리 결과와 히스토그램을 만들고 판정 결과를 집계합니다."""
import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2

from scanner.constants import SUPPORTED_IMAGE_SUFFIXES
from scanner.io import read_image, save_image, save_scan
from scanner.pipeline import Parameters, scan
from tools.evaluation_analysis import analyze_image, histogram_plot
from tools.evaluation_report import summarize

GROUPS = ('simple', 'shadow', 'tilted', 'complex')
IMAGES_PER_GROUP = 5
REQUIRED_IMAGE_COUNT = len(GROUPS) * IMAGES_PER_GROUP
DEFAULT_OUTPUT = Path('outputs/evaluation')
DEFAULT_MANIFEST = Path('data')
PRESETS = {
    'default': Parameters(),
    'sensitive': Parameters(blur=3, low=10, high=40),
    'strict': Parameters(blur=9, low=100, high=220),
}


def _load_manifest(path: Union[Path, str]) -> Dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        image_paths = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if len(image_paths) != REQUIRED_IMAGE_COUNT:
            raise ValueError(
                f'Personal image directory requires exactly '
                f'{REQUIRED_IMAGE_COUNT} PNG or JPEG images'
            )
        return {
            'provenance': 'personal',
            'images': [
                {
                    'id': image.stem,
                    'path': image.name,
                    'condition': image.stem.split('_', 1)[0],
                    'notes': 'User-provided image; record lighting, angle and background',
                    '_path': image.resolve(),
                }
                for image in image_paths
            ],
        }

    manifest = json.loads(path.read_text(encoding='utf-8'))
    cases = manifest.get('images', [])
    if len(cases) != REQUIRED_IMAGE_COUNT or any(
        sum(c.get('condition') == group for c in cases) != IMAGES_PER_GROUP
        for group in GROUPS
    ):
        raise ValueError(
            f'Manifest requires {REQUIRED_IMAGE_COUNT} images, '
            f'exactly {IMAGES_PER_GROUP} per condition'
        )

    ids, paths = set(), set()
    for case in cases:
        if case['id'] in ids or case['path'] in paths:
            raise ValueError('Image IDs and paths must be unique')
        if not case.get('notes'):
            raise ValueError('Each image needs capture-condition notes')
        ids.add(case['id'])
        paths.add(case['path'])

        image_path = (path.parent / case['path']).resolve()
        case['_path'] = image_path
    return manifest


def evaluate(
    manifest_path: Union[Path, str],
    output: Path,
) -> List[Dict[str, Any]]:
    if (output / 'results.csv').exists():
        raise ValueError('results.csv already exists; use --summarize or a new output directory')
    manifest = _load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    rows, eda = [], []
    histograms = defaultdict(list)

    for index, case in enumerate(manifest['images']):
        image = read_image(case['_path'])
        case_output = output / f"{index + 1:02d}_{case['id']}"
        statistics, hist = analyze_image(image)
        histograms[case['condition']].append(hist)
        save_image(
            case_output / 'histogram.png',
            histogram_plot({case['id']: [hist]}),
        )
        eda.append({
            'id': case['id'],
            'condition': case['condition'],
            **statistics,
            'histogram': hist.tolist(),
        })

        for preset, params in PRESETS.items():
            result = scan(image, params)
            rows.append({
                'id': case['id'],
                'condition': case['condition'],
                'preset': preset,
                'detected': result.corners is not None,
                'success': '',
                'review_notes': '',
                'notes': case['notes'],
            })

            save_scan(result, case_output / preset)

    with (output / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (output / 'eda.json').write_text(
        json.dumps(eda, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    (output / 'parameters.json').write_text(
        json.dumps({k: asdict(v) for k, v in PRESETS.items()}, indent=2),
        encoding='utf-8',
    )
    save_image(output / 'histograms.png', histogram_plot(histograms))

    summarize(output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', nargs='?', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--summarize', action='store_true',
                        help='Summarize manually reviewed results.csv without processing images')
    args = parser.parse_args()

    try:
        if args.summarize:
            summarize(args.output)
            return
        evaluate(args.manifest, args.output)
    except (ValueError, KeyError, OSError, cv2.error) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
