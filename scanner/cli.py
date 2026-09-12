"""Image and webcam entry points with optional OpenCV desktop controls."""
import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .io import read_image, save_scan
from .pipeline import Parameters, scan


def show(result):
    stages = dict(result.stages)
    if result.corners is None:
        blank = np.zeros((160, 480, 3), dtype=np.uint8)
        cv2.putText(blank, 'No document found', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        stages.update({'05_warped': blank, '06_result': blank})
    for name, image in stages.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)


def interactive(params, output, image=None, camera=None):
    if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise ValueError('No desktop display. Use --headless for image processing.')
    capture = None
    try:
        if camera is not None:
            capture = cv2.VideoCapture(camera)
            if not capture.isOpened():
                raise ValueError('Cannot open camera {}'.format(camera))
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Blur radius', 'Controls', params.blur // 2, 25, lambda _: None)
        cv2.createTrackbar('Canny low', 'Controls', params.low, 254, lambda _: None)
        cv2.createTrackbar('Canny high', 'Controls', params.high, 255, lambda _: None)
        previous = None
        result = None
        saved = 0
        while True:
            low = cv2.getTrackbarPos('Canny low', 'Controls')
            high = max(low + 1, cv2.getTrackbarPos('Canny high', 'Controls'))
            current = replace(params, blur=2 * cv2.getTrackbarPos('Blur radius', 'Controls') + 1, low=low, high=high)
            if capture is not None:
                ok, image = capture.read()
                if not ok:
                    raise ValueError('Camera frame could not be read')
            if capture is not None or current != previous:
                result = scan(image, current)
                show(result)
                previous = current
            key = cv2.waitKey(30) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty('Controls', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('s'):
                if result.corners is None:
                    print('No document found; adjust parameters before saving.', file=sys.stderr)
                    continue
                destination = output if capture is None else output / ('capture_{:03d}'.format(saved))
                save_scan(result, destination)
                print('Saved: {}'.format(destination.resolve()))
                saved += 1
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Rectify a document photo. GUI: s=save, q/Esc=quit.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--webcam', type=int, metavar='INDEX')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/scan'))
    parser.add_argument('--blur', type=int, default=5)
    parser.add_argument('--low', type=int, default=50)
    parser.add_argument('--high', type=int, default=150)
    parser.add_argument('--min-area', type=float, default=0.08)
    parser.add_argument('--epsilon', type=float, default=0.02)
    parser.add_argument('--block-size', type=int, default=31)
    parser.add_argument('--threshold-c', type=float, default=10)
    args = parser.parse_args(argv)
    try:
        params = Parameters(args.blur, args.low, args.high, args.min_area, args.epsilon, args.block_size, args.threshold_c)
        if args.webcam is not None:
            if args.headless:
                raise ValueError('--webcam requires a desktop display; remove --headless')
            interactive(params, args.output, camera=args.webcam)
        else:
            image = read_image(args.image)
            if args.headless:
                result = scan(image, params)
                save_scan(result, args.output)
                if result.corners is None:
                    raise ValueError('No document quadrilateral found. Diagnostic stages saved; adjust parameters.')
                print('Saved: {}'.format(args.output.resolve()))
            else:
                interactive(params, args.output, image=image)
        return 0
    except (ValueError, OSError, cv2.error) as error:
        print('Error: {}'.format(error), file=sys.stderr)
        return 2
