"""명령행 옵션을 읽고 GUI 또는 파일 처리를 실행합니다."""
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2

from .gui import interactive
from .io import read_image, save_scan
from .pipeline import Parameters, scan

DEFAULT_OUTPUT = Path('outputs/scan')
DEFAULT_IMAGE = Path('data/simple_1.png')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Rectify a document photo. GUI: s=save, q/Esc=quit.',
    )
    parser.add_argument('--image', type=Path, default=DEFAULT_IMAGE)
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

        image = read_image(args.image)
        if args.headless:
            result = scan(image, params)
            save_scan(result, args.output)
            if result.corners is None:
                raise ValueError(
                    'No document quadrilateral found. '
                    'Diagnostic stages saved; adjust parameters.'
                )
            print(f'Saved: {args.output.resolve()}')
        else:
            interactive(params, args.output, image=image)
        return 0
    except (ValueError, OSError, cv2.error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2
