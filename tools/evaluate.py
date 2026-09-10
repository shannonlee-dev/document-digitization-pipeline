"""Reproducible corner evaluation and histogram EDA; no OCR or learned models."""
import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from scanner.pipeline import Parameters, order_corners, read_image, save_image, save_scan, scan

GROUPS = ('simple', 'shadow', 'tilted', 'complex')
PRESETS = {
    'default': Parameters(),
    'sensitive': Parameters(blur=3, low=10, high=40),
    'strict': Parameters(blur=9, low=100, high=220),
}


def corner_error(predicted, expected, shape):
    if predicted is None:
        return None
    predicted, expected = order_corners(predicted), order_corners(expected)
    # Cyclic matching removes arbitrary start-vertex choices for diamonds.
    distances = [np.max(np.linalg.norm(predicted - np.roll(expected, k, axis=0), axis=1)) for k in range(4)]
    return float(min(distances) / np.hypot(*shape[:2]))


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('provenance') not in ('synthetic', 'personal', 'lms'):
        raise ValueError('provenance must be synthetic, personal, or lms')
    cases = manifest.get('images', [])
    if len(cases) != 20 or any(sum(c.get('condition') == group for c in cases) != 5 for group in GROUPS):
        raise ValueError('Manifest requires 20 images, exactly 5 per condition')
    ids, paths = set(), set()
    for case in cases:
        if case['id'] in ids or case['path'] in paths:
            raise ValueError('Image IDs and paths must be unique')
        if not case.get('notes'):
            raise ValueError('Each image needs capture-condition notes')
        ids.add(case['id'])
        paths.add(case['path'])
        corners = order_corners(case['corners'])
        image_path = (path.parent / case['path']).resolve()
        case['_path'] = image_path
        image = read_image(image_path)
        if np.any(corners < 0) or np.any(corners[:, 0] >= image.shape[1]) or np.any(corners[:, 1] >= image.shape[0]):
            raise ValueError('Ground-truth corners lie outside image: {}'.format(case['id']))
    return manifest


def histogram_plot(histograms):
    canvas = np.full((400, 760, 3), 255, np.uint8)
    colors = ((50, 130, 20), (180, 80, 0), (0, 90, 220), (150, 0, 140))
    for index, (group, values) in enumerate(histograms.items()):
        histogram = np.mean(values, axis=0)
        # Log scale shows both background peaks and weak tails.
        y = np.log1p(histogram * 10000) / np.log1p(10000)
        points = np.column_stack((40 + np.arange(256) * 2.6, 340 - y * 260)).astype(np.int32)
        cv2.polylines(canvas, [points], False, colors[index], 2)
        cv2.putText(canvas, group, (45 + index * 175, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, colors[index], 1)
    cv2.putText(canvas, 'Grayscale 0 -> 255; log(1 + probability * 10000)', (40, 380), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    return canvas


def stage_preview(result):
    tiles = []
    for name in ('01_original', '02_preprocessed', '03_edges', '04_contours', '05_warped', '06_result'):
        tile = np.full((250, 320, 3), 245, np.uint8)
        image = result.stages.get(name)
        if image is not None:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            scale = min(320 / image.shape[1], 220 / image.shape[0])
            image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
            tile[30:30 + image.shape[0], :image.shape[1]] = image
        cv2.putText(tile, name if image is not None else name + ': missing', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        tiles.append(tile)
    return np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))


def evaluate(manifest_path, output, tolerance=0.03):
    if not 0 < tolerance < 1:
        raise ValueError('tolerance must be in (0, 1)')
    manifest = load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    rows, eda = [], []
    histograms = defaultdict(list)
    for index, case in enumerate(manifest['images']):
        image = read_image(case['_path'])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = np.bincount(gray.ravel(), minlength=256) / gray.size
        histograms[case['condition']].append(hist)
        save_image(output / '{:02d}'.format(index + 1) / 'histogram.png', histogram_plot({case['id']: [hist]}))
        eda.append({'id': case['id'], 'condition': case['condition'], 'mean': float(gray.mean()), 'std': float(gray.std()), 'p05': float(np.percentile(gray, 5)), 'p95': float(np.percentile(gray, 95)), 'histogram': hist.tolist()})
        for preset, params in PRESETS.items():
            result = scan(image, params)
            error = corner_error(result.corners, case['corners'], image.shape)
            success = error is not None and error <= tolerance and '06_result' in result.stages
            rows.append({'id': case['id'], 'condition': case['condition'], 'preset': preset, 'detected': result.corners is not None, 'success': success, 'max_corner_error': error, 'notes': case['notes']})
            save_scan(result, output / '{:02d}'.format(index + 1) / preset)
            save_image(output / '{:02d}'.format(index + 1) / preset / 'preview.jpg', stage_preview(result))
    with (output / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (output / 'eda.json').write_text(json.dumps(eda, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'parameters.json').write_text(json.dumps({k: asdict(v) for k, v in PRESETS.items()}, indent=2), encoding='utf-8')
    save_image(output / 'histograms.png', histogram_plot(histograms))
    lines = ['# Evaluation results', '', 'Provenance: **{}**'.format(manifest['provenance']), '',
             'Automatic geometric proxy: maximum matched corner distance / image diagonal <= {:.3f}, and warp produced.'.format(tolerance),
             'This does not certify text quality or physical aspect ratio. Inspect saved warps before final submission.', '',
             '| Preset | Condition | Total | Success | Failure | Rate |', '|---|---|---:|---:|---:|---:|']
    for preset in PRESETS:
        for group in GROUPS:
            subset = [r for r in rows if r['preset'] == preset and r['condition'] == group]
            successes = sum(r['success'] for r in subset)
            lines.append('| {} | {} | {} | {} | {} | {:.0%} |'.format(preset, group, len(subset), successes, len(subset) - successes, successes / len(subset)))
    lines.extend(['', '## Per-image parameter comparison', '', '| Image | Condition | default | sensitive | strict |', '|---|---|---|---|---|'])
    for case in manifest['images']:
        values = [next(r for r in rows if r['id'] == case['id'] and r['preset'] == p) for p in PRESETS]
        cells = ['{} ({})'.format('PASS' if r['success'] else 'FAIL', 'not detected' if r['max_corner_error'] is None else '{:.4f}'.format(r['max_corner_error'])) for r in values]
        lines.append('| {} | {} | {} |'.format(case['id'], case['condition'], ' | '.join(cells)))
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation'))
    parser.add_argument('--tolerance', type=float, default=0.03)
    parser.add_argument('--epsilon-sweep', action='store_true', help='Also compare approximation ratios 0.01, 0.02, 0.04, 0.06')
    args = parser.parse_args()
    try:
        evaluate(args.manifest, args.output, args.tolerance)
        if args.epsilon_sweep:
            rows = []
            manifest = load_manifest(args.manifest)
            for epsilon in (0.01, 0.02, 0.04, 0.06):
                for case in manifest['images']:
                    image = read_image(case['_path'])
                    result = scan(image, Parameters(epsilon=epsilon))
                    error = corner_error(result.corners, case['corners'], image.shape)
                    rows.append({'id': case['id'], 'epsilon': epsilon, 'success': error is not None and error <= args.tolerance, 'error': error})
            with (args.output / 'epsilon_experiment.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
    except (ValueError, KeyError, OSError, cv2.error) as error:
        parser.exit(2, 'Error: {}\n'.format(error))


if __name__ == '__main__':
    main()
