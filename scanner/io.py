"""이미지를 읽고 스캔 단계를 PNG/JPEG 파일로 저장합니다."""
from pathlib import Path
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np

from .constants import (
    MIN_IMAGE_SIZE,
    OUTPUT_STAGES,
    STAGE_IMAGE_SUFFIX,
    SUPPORTED_IMAGE_SUFFIXES,
)

if TYPE_CHECKING:
    from .pipeline import Scan


def read_image(path: Union[Path, str]) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Image does not exist: {path}')
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < MIN_IMAGE_SIZE:
        raise ValueError(f'Not a readable image: {path}')
    return image


def save_image(path: Union[Path, str], image: np.ndarray) -> None:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError('Output must be PNG or JPEG')

    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'Image encoding failed: {path}')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f'Image save failed: {path}')


def save_scan(result: 'Scan', output: Union[Path, str]) -> None:
    output = Path(output)
    for name, image in result.stages.items():
        save_image(output / (name + STAGE_IMAGE_SUFFIX), image)

    # 재실행이 실패했을 때 이전 결과가 남아 성공처럼 보이지 않게 합니다.
    for name in OUTPUT_STAGES:
        stale = output / (name + STAGE_IMAGE_SUFFIX)
        if name not in result.stages and stale.exists():
            stale.unlink()
