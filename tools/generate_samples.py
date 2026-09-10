"""Generate deterministic synthetic fixtures, never substitutes for LMS photos."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scanner.pipeline import save_image


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(19)
    cases = []
    for group in ('simple', 'shadow', 'tilted', 'complex'):
        for index in range(5):
            image = np.full((600, 800, 3), 55, np.uint8)
            corners = np.float32([[220, 75], [590, 90], [610, 525], [200, 510]])
            if group == 'simple':
                variants = [
                    [[210, 65], [590, 65], [590, 525], [210, 525]],
                    [[300, 55], [490, 60], [495, 535], [295, 530]],
                    [[170, 185], [630, 180], [635, 410], [165, 415]],
                    [[180, 85], [565, 110], [600, 510], [195, 495]],
                    [[240, 100], [600, 70], [570, 520], [205, 500]],
                ]
                corners = np.float32(variants[index])
            notes = 'uniform lighting; plain background; near frontal'
            if group == 'tilted':
                angle = np.deg2rad(46 + index * 7)
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                corners = ((np.float32([[-145, -210], [145, -210], [145, 210], [-145, 210]]) @ rotation.T) + [400, 300]).astype(np.float32)
                notes = 'in-plane rotation {} degrees, not calibrated camera tilt'.format(46 + index * 7)
            if group == 'complex':
                noise = rng.integers(20, 130, image.shape[:2], dtype=np.uint8)
                if index < 2:
                    noise = cv2.GaussianBlur(noise, (15, 15), 0)
                image[:] = noise[:, :, None]
                for _ in range(8 if index < 2 else 25):
                    p = rng.integers([0, 0], [800, 600], size=(2, 2))
                    cv2.line(image, tuple(p[0]), tuple(p[1]), (110, 110, 110), 2)
                notes = 'cluttered background with random lines'
            paper = np.full((450, 380, 3), 235 - index * 3, np.uint8)
            cv2.putText(paper, 'SAMPLE {} {}'.format(group.upper(), index + 1), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1)
            for y in range(55, 420, 32):
                cv2.putText(paper, 'DOCUMENT line {}'.format(y), (20, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 40, 40), 1)
            matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [379, 0], [379, 449], [0, 449]]), corners)
            warped = cv2.warpPerspective(paper, matrix, (800, 600))
            mask = cv2.warpPerspective(np.full((450, 380), 255, np.uint8), matrix, (800, 600))
            image[mask > 0] = warped[mask > 0]
            if group == 'shadow':
                factors = np.ones((600, 800), np.float32)
                factors[:, :400] = [0.65, .4, .15, .08, .03][index]
                image = (image * factors[:, :, None]).astype(np.uint8)
                notes = 'hard shadow over left half; brightness factor {}'.format([.65, .4, .15, .08, .03][index])
            if group == 'complex' and index >= 2:
                cv2.rectangle(image, (25, 25), (775, 575), (245, 245, 245), 8)
                notes += '; larger bright rectangular distractor'
            name = '{}_{}'.format(group, index + 1)
            save_image(output / (name + '.png'), image)
            cases.append({'id': name, 'path': name + '.png', 'condition': group, 'notes': notes, 'corners': corners.tolist()})
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps({'provenance': 'synthetic', 'images': cases}, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/synthetic'))
    generate(parser.parse_args().output)
