"""수동 평가용 처리 결과와 히스토그램을 만들고 판정 결과를 집계합니다."""
import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

import cv2
import numpy as np

from scanner.constants import (
    GRAYSCALE_LEVELS,
    MAX_PIXEL_VALUE,
    SCAN_STAGES,
    SUPPORTED_IMAGE_SUFFIXES,
)
from scanner.io import read_image, save_image, save_scan
from scanner.pipeline import Parameters, Scan, scan

GROUPS = ('simple', 'shadow', 'tilted', 'complex')
IMAGES_PER_GROUP = 5
REQUIRED_IMAGE_COUNT = len(GROUPS) * IMAGES_PER_GROUP
DEFAULT_OUTPUT = Path('outputs/evaluation')
DEFAULT_MANIFEST = Path('data')
EPSILON_SWEEP = (0.01, 0.02, 0.04, 0.06)
BRIGHTNESS_PERCENTILES = (5, 95)

HISTOGRAM_SHAPE = (400, 760, 3)
HISTOGRAM_COLORS = ((50, 130, 20), (180, 80, 0), (0, 90, 220), (150, 0, 140))
HISTOGRAM_LOG_SCALE = 10000
HISTOGRAM_X_ORIGIN = 40
HISTOGRAM_X_SCALE = 2.6
HISTOGRAM_Y_BASELINE = 340
HISTOGRAM_HEIGHT = 260
HISTOGRAM_LINE_THICKNESS = 2
HISTOGRAM_LEGEND_POSITION = (45, 30)
HISTOGRAM_LEGEND_SPACING = 175
HISTOGRAM_CAPTION_POSITION = (40, 380)

PREVIEW_TILE_WIDTH = 320
PREVIEW_TILE_HEIGHT = 250
PREVIEW_HEADER_HEIGHT = 30
PREVIEW_BACKGROUND = 245
PREVIEW_COLUMNS = 3
PREVIEW_LABEL_POSITION = (8, 20)
PLOT_TEXT_SCALE = 0.5
PLOT_TEXT_COLOR = (0, 0, 0)
PLOT_TEXT_THICKNESS = 1
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


def _histogram_plot(histograms: Mapping[str, Sequence[np.ndarray]]) -> np.ndarray:
    canvas = np.full(HISTOGRAM_SHAPE, MAX_PIXEL_VALUE, np.uint8)
    for index, (group, values) in enumerate(histograms.items()):
        histogram = np.mean(values, axis=0)
        # 로그 스케일로 배경 피크와 약한 꼬리를 함께 보여줍니다.
        y = np.log1p(histogram * HISTOGRAM_LOG_SCALE) / np.log1p(HISTOGRAM_LOG_SCALE)
        points = np.column_stack((
            HISTOGRAM_X_ORIGIN + np.arange(GRAYSCALE_LEVELS) * HISTOGRAM_X_SCALE,
            HISTOGRAM_Y_BASELINE - y * HISTOGRAM_HEIGHT,
        )).astype(np.int32)
        cv2.polylines(
            canvas, [points], False,
            HISTOGRAM_COLORS[index], HISTOGRAM_LINE_THICKNESS,
        )
        cv2.putText(
            canvas, group,
            (
                HISTOGRAM_LEGEND_POSITION[0] + index * HISTOGRAM_LEGEND_SPACING,
                HISTOGRAM_LEGEND_POSITION[1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX, PLOT_TEXT_SCALE,
            HISTOGRAM_COLORS[index], PLOT_TEXT_THICKNESS,
        )

    cv2.putText(
        canvas,
        f'Grayscale 0 -> {MAX_PIXEL_VALUE}; '
        f'log(1 + probability * {HISTOGRAM_LOG_SCALE})',
        HISTOGRAM_CAPTION_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
        PLOT_TEXT_SCALE, PLOT_TEXT_COLOR, PLOT_TEXT_THICKNESS,
    )
    return canvas


def _stage_preview(result: Scan) -> np.ndarray:
    tiles = []
    for name in SCAN_STAGES:
        tile = np.full(
            (PREVIEW_TILE_HEIGHT, PREVIEW_TILE_WIDTH, 3),
            PREVIEW_BACKGROUND, np.uint8,
        )
        image = result.stages.get(name)
        if image is not None:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            scale = min(
                PREVIEW_TILE_WIDTH / image.shape[1],
                (PREVIEW_TILE_HEIGHT - PREVIEW_HEADER_HEIGHT) / image.shape[0],
            )
            image = cv2.resize(image, (
                max(1, round(image.shape[1] * scale)),
                max(1, round(image.shape[0] * scale)),
            ))
            tile[
                PREVIEW_HEADER_HEIGHT:PREVIEW_HEADER_HEIGHT + image.shape[0],
                :image.shape[1],
            ] = image

        cv2.putText(
            tile, name if image is not None else name + ': missing', PREVIEW_LABEL_POSITION,
            cv2.FONT_HERSHEY_SIMPLEX, PLOT_TEXT_SCALE, PLOT_TEXT_COLOR, PLOT_TEXT_THICKNESS,
        )
        tiles.append(tile)

    return np.vstack((
        np.hstack(tiles[:PREVIEW_COLUMNS]),
        np.hstack(tiles[PREVIEW_COLUMNS:]),
    ))


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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = np.bincount(gray.ravel(), minlength=GRAYSCALE_LEVELS) / gray.size
        histograms[case['condition']].append(hist)
        save_image(
            case_output / 'histogram.png',
            _histogram_plot({case['id']: [hist]}),
        )
        eda.append({
            'id': case['id'],
            'condition': case['condition'],
            'mean': float(gray.mean()),
            'std': float(gray.std()),
            'p05': float(np.percentile(gray, BRIGHTNESS_PERCENTILES[0])),
            'p95': float(np.percentile(gray, BRIGHTNESS_PERCENTILES[1])),
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
            save_image(
                case_output / preset / 'preview.jpg',
                _stage_preview(result),
            )

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
    save_image(output / 'histograms.png', _histogram_plot(histograms))

    summarize(output)
    return rows


def summarize(output: Path) -> None:
    """수동 판정 CSV를 집계하며 미판정은 성공·실패에 포함하지 않습니다."""
    with (output / 'results.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('results.csv is empty')
    for row in rows:
        row['success'] = row['success'].strip().lower()
        if row['success'] not in ('', 'true', 'false'):
            raise ValueError('success must be true, false, or blank')

    lines = [
        '# Evaluation results', '',
        'Manual review: inspect 04_contours.png, 05_warped.png and 06_result.png.',
        'PASS requires four correct document corners and a straightened document.',
        'Fill success with true/false in results.csv; leave unreviewed rows blank.',
        'Record failure reasons and parameter adjustment results in review_notes.',
        'Analyze at least three failures. Rates appear only after every row in a group is reviewed.',
        '',
        '| Preset | Condition | Total | Success | Failure | Pending | Rate |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for preset in dict.fromkeys(r['preset'] for r in rows):
        preset_rows = [r for r in rows if r['preset'] == preset]
        for group in sorted({r['condition'] for r in preset_rows}) + ['ALL']:
            subset = [r for r in preset_rows if group == 'ALL' or r['condition'] == group]
            successes = sum(r['success'] == 'true' for r in subset)
            failures = sum(r['success'] == 'false' for r in subset)
            pending = len(subset) - successes - failures
            rate = 'pending' if pending else f'{successes / len(subset):.0%}'
            lines.append(
                f'| {preset} | {group} | {len(subset)} | {successes} | '
                f'{failures} | {pending} | {rate} |'
            )
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', nargs='?', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--summarize', action='store_true',
                        help='Summarize manually reviewed results.csv without processing images')
    parser.add_argument(
        '--epsilon-sweep', action='store_true',
        help='Also compare approximation ratios ' + ', '.join(map(str, EPSILON_SWEEP)),
    )
    args = parser.parse_args()

    try:
        if args.summarize:
            summarize(args.output)
            return
        evaluate(args.manifest, args.output)
        if args.epsilon_sweep:
            rows = []
            manifest = _load_manifest(args.manifest)
            for epsilon in EPSILON_SWEEP:
                for case in manifest['images']:
                    image = read_image(case['_path'])
                    result = scan(image, Parameters(epsilon=epsilon))
                    save_scan(result, args.output / case['id'] / f'epsilon_{epsilon}')
                    rows.append({
                        'id': case['id'],
                        'epsilon': epsilon,
                        'detected': result.corners is not None,
                        'success': '',
                        'review_notes': '',
                    })

            with (args.output / 'epsilon_experiment.csv').open(
                'w', newline='', encoding='utf-8',
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(rows[0]), lineterminator='\n',
                )
                writer.writeheader()
                writer.writerows(rows)
    except (ValueError, KeyError, OSError, cv2.error) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
