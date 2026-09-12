"""OCR이나 학습 모델 없이 꼭짓점 평가와 히스토그램 분석을 재현합니다."""
import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from numpy.typing import ArrayLike

from scanner.constants import (
    DOCUMENT_CORNER_COUNT,
    GRAYSCALE_LEVELS,
    MAX_PIXEL_VALUE,
    SCAN_STAGES,
    SUPPORTED_IMAGE_SUFFIXES,
)
from scanner.io import read_image, save_image, save_scan
from scanner.pipeline import Parameters, Scan, _order_corners, scan

GROUPS = ('simple', 'shadow', 'tilted', 'complex')
IMAGES_PER_GROUP = 5
REQUIRED_IMAGE_COUNT = len(GROUPS) * IMAGES_PER_GROUP
SUPPORTED_PROVENANCES = ('synthetic', 'personal', 'lms')
DEFAULT_OUTPUT = Path('outputs/evaluation')
DEFAULT_TOLERANCE = 0.03
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

def _corner_error(
    predicted: Optional[ArrayLike],
    expected: ArrayLike,
    shape: Tuple[int, ...],
) -> Optional[float]:
    if predicted is None:
        return None

    predicted, expected = _order_corners(predicted), _order_corners(expected)
    # 순환 매칭으로 마름모의 임의 시작점 차이를 제거합니다.
    distances = [
        np.max(np.linalg.norm(predicted - np.roll(expected, k, axis=0), axis=1))
        for k in range(DOCUMENT_CORNER_COUNT)
    ]
    return float(min(distances) / np.hypot(*shape[:2]))


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
                    'condition': 'personal',
                    'notes': 'User-provided screenshot; ground-truth corners not supplied',
                    '_path': image.resolve(),
                }
                for image in image_paths
            ],
        }

    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('provenance') not in SUPPORTED_PROVENANCES:
        raise ValueError('provenance must be synthetic, personal, or lms')
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
        image = read_image(image_path)
        if 'corners' in case:
            corners = _order_corners(case['corners'])
            if (
                np.any(corners < 0)
                or np.any(corners[:, 0] >= image.shape[1])
                or np.any(corners[:, 1] >= image.shape[0])
            ):
                raise ValueError(
                    f"Ground-truth corners lie outside image: {case['id']}"
                )
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
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[Dict[str, Any]]:
    if not 0 < tolerance < 1:
        raise ValueError('tolerance must be in (0, 1)')

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
            error = (
                _corner_error(result.corners, case['corners'], image.shape)
                if 'corners' in case else None
            )
            success = error is not None and error <= tolerance and '06_result' in result.stages
            if 'corners' not in case:
                success = result.corners is not None and '06_result' in result.stages
            rows.append({
                'id': case['id'],
                'condition': case['condition'],
                'preset': preset,
                'detected': result.corners is not None,
                'success': success,
                'max_corner_error': error,
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

    has_ground_truth = all('corners' in case for case in manifest['images'])
    metric = (
        'Automatic geometric proxy: maximum matched corner distance / '
        f'image diagonal <= {tolerance:.3f}, and warp produced.'
        if has_ground_truth else
        'Personal-image proxy: document quadrilateral detected and warp produced.'
    )
    lines = [
        '# Evaluation results',
        '',
        f"Provenance: **{manifest['provenance']}**",
        '',
        metric,
        'This does not certify text quality or physical aspect ratio. '
        'Inspect saved warps before final submission.',
        '',
        '| Preset | Condition | Total | Success | Failure | Rate |',
        '|---|---|---:|---:|---:|---:|',
    ]
    groups = sorted({case['condition'] for case in manifest['images']})
    for preset in PRESETS:
        for group in groups:
            subset = [r for r in rows if r['preset'] == preset and r['condition'] == group]
            successes = sum(r['success'] for r in subset)
            lines.append(
                f'| {preset} | {group} | {len(subset)} | {successes} | '
                f'{len(subset) - successes} | {successes / len(subset):.0%} |'
            )

    lines.extend([
        '',
        '## Per-image parameter comparison',
        '',
        '| Image | Condition | default | sensitive | strict |',
        '|---|---|---|---|---|',
    ])
    for case in manifest['images']:
        values = [
            next(r for r in rows if r['id'] == case['id'] and r['preset'] == p)
            for p in PRESETS
        ]
        cells = []
        for result in values:
            detail = (
                ('detected' if result['detected'] else 'not detected')
                if not has_ground_truth else (
                    'not detected' if result['max_corner_error'] is None
                    else f"{result['max_corner_error']:.4f}"
                )
            )
            cells.append(f"{'PASS' if result['success'] else 'FAIL'} ({detail})")
        lines.append(f"| {case['id']} | {case['condition']} | {' | '.join(cells)} |")

    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        '--epsilon-sweep', action='store_true',
        help='Also compare approximation ratios ' + ', '.join(map(str, EPSILON_SWEEP)),
    )
    args = parser.parse_args()

    try:
        evaluate(args.manifest, args.output, args.tolerance)
        if args.epsilon_sweep:
            rows = []
            manifest = _load_manifest(args.manifest)
            for epsilon in EPSILON_SWEEP:
                for case in manifest['images']:
                    image = read_image(case['_path'])
                    result = scan(image, Parameters(epsilon=epsilon))
                    error = (
                        _corner_error(result.corners, case['corners'], image.shape)
                        if 'corners' in case else None
                    )
                    success = result.corners is not None and '06_result' in result.stages
                    if error is not None:
                        success = error <= args.tolerance and '06_result' in result.stages
                    rows.append({
                        'id': case['id'],
                        'epsilon': epsilon,
                        'success': success,
                        'error': error,
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
