"""이미지·웹캠 진입점과 OpenCV 데스크톱 제어 화면을 제공합니다."""
import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .constants import MAX_BLUR_SIZE, MAX_PIXEL_VALUE, OUTPUT_STAGES
from .io import read_image, save_scan
from .pipeline import Image, Parameters, Scan, scan

DEFAULT_OUTPUT = Path('outputs/scan')
CONTROLS_WINDOW = 'Controls'
BLUR_TRACKBAR = 'Blur radius'
CANNY_LOW_TRACKBAR = 'Canny low'
CANNY_HIGH_TRACKBAR = 'Canny high'
FRAME_DELAY_MS = 30
KEY_CODE_MASK = 0xff
QUIT_KEYS = (ord('q'), 27)
SAVE_KEY = ord('s')
MISSING_DOCUMENT_SHAPE = (160, 480, 3)
MISSING_DOCUMENT_TEXT_POSITION = (20, 80)
MISSING_DOCUMENT_TEXT_SCALE = 0.8
MISSING_DOCUMENT_TEXT_COLOR = (255, 255, 255)
MISSING_DOCUMENT_TEXT_THICKNESS = 2


def _show(result: Scan) -> None:
    stages = dict(result.stages)
    if result.corners is None:
        blank = np.zeros(MISSING_DOCUMENT_SHAPE, dtype=np.uint8)
        cv2.putText(
            blank, 'No document found', MISSING_DOCUMENT_TEXT_POSITION,
            cv2.FONT_HERSHEY_SIMPLEX, MISSING_DOCUMENT_TEXT_SCALE,
            MISSING_DOCUMENT_TEXT_COLOR, MISSING_DOCUMENT_TEXT_THICKNESS,
        )
        stages.update({name: blank for name in OUTPUT_STAGES})

    for name, image in stages.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)


def _interactive(
    params: Parameters,
    output: Path,
    image: Optional[Image] = None,
    camera: Optional[int] = None,
) -> None:
    if sys.platform.startswith('linux') and not (
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    ):
        raise ValueError('No desktop display. Use --headless for image processing.')

    capture = None
    try:
        if camera is not None:
            capture = cv2.VideoCapture(camera)
            if not capture.isOpened():
                raise ValueError('Cannot open camera {}'.format(camera))

        cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            BLUR_TRACKBAR, CONTROLS_WINDOW,
            params.blur // 2, MAX_BLUR_SIZE // 2, lambda _: None,
        )
        cv2.createTrackbar(
            CANNY_LOW_TRACKBAR, CONTROLS_WINDOW,
            params.low, MAX_PIXEL_VALUE - 1, lambda _: None,
        )
        cv2.createTrackbar(
            CANNY_HIGH_TRACKBAR, CONTROLS_WINDOW,
            params.high, MAX_PIXEL_VALUE, lambda _: None,
        )

        previous = None
        result = None
        saved = 0

        while True:
            low = cv2.getTrackbarPos(CANNY_LOW_TRACKBAR, CONTROLS_WINDOW)
            high = max(low + 1, cv2.getTrackbarPos(CANNY_HIGH_TRACKBAR, CONTROLS_WINDOW))
            current = replace(
                params,
                blur=2 * cv2.getTrackbarPos(BLUR_TRACKBAR, CONTROLS_WINDOW) + 1,
                low=low,
                high=high,
            )

            if capture is not None:
                ok, image = capture.read()
                if not ok:
                    raise ValueError('Camera frame could not be read')

            if capture is not None or current != previous:
                result = scan(image, current)
                _show(result)
                previous = current

            key = cv2.waitKey(FRAME_DELAY_MS) & KEY_CODE_MASK
            if (
                key in QUIT_KEYS
                or cv2.getWindowProperty(CONTROLS_WINDOW, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            if key == SAVE_KEY:
                if result.corners is None:
                    print(
                        'No document found; adjust parameters before saving.',
                        file=sys.stderr,
                    )
                    continue

                destination = (
                    output if capture is None
                    else output / ('capture_{:03d}'.format(saved))
                )
                save_scan(result, destination)
                print('Saved: {}'.format(destination.resolve()))
                saved += 1
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Rectify a document photo. GUI: s=save, q/Esc=quit.',
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--webcam', type=int, metavar='INDEX')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--blur', type=int, default=Parameters.blur)
    parser.add_argument('--low', type=int, default=Parameters.low)
    parser.add_argument('--high', type=int, default=Parameters.high)
    parser.add_argument('--min-area', type=float, default=Parameters.min_area)
    parser.add_argument('--epsilon', type=float, default=Parameters.epsilon)
    parser.add_argument('--block-size', type=int, default=Parameters.block_size)
    parser.add_argument('--threshold-c', type=float, default=Parameters.threshold_c)
    args = parser.parse_args(argv)

    try:
        params = Parameters(
            args.blur, args.low, args.high,
            args.min_area, args.epsilon,
            args.block_size, args.threshold_c,
        )

        if args.webcam is not None:
            if args.headless:
                raise ValueError('--webcam requires a desktop display; remove --headless')
            _interactive(params, args.output, camera=args.webcam)
        else:
            image = read_image(args.image)
            if args.headless:
                result = scan(image, params)
                save_scan(result, args.output)
                if result.corners is None:
                    raise ValueError(
                        'No document quadrilateral found. '
                        'Diagnostic stages saved; adjust parameters.'
                    )
                print('Saved: {}'.format(args.output.resolve()))
            else:
                _interactive(params, args.output, image=image)
        return 0
    except (ValueError, OSError, cv2.error) as error:
        print('Error: {}'.format(error), file=sys.stderr)
        return 2
