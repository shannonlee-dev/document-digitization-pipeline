"""이미지를 읽고 스캔 단계를 PNG/JPEG 파일로 저장합니다."""
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Image does not exist: {path}')
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 3:
        raise ValueError(f'Not a readable image: {path}')
    return image


def save_image(path, image):
    path = Path(path)
    if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
        raise ValueError('Output must be PNG or JPEG')
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError('Image encoding failed: {}'.format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError('Image save failed: {}'.format(path))


def save_scan(result, output):
    output = Path(output)
    for name, image in result.stages.items():
        save_image(output / (name + '.png'), image)
    # 재실행이 실패했을 때 이전 결과가 남아 성공처럼 보이지 않게 합니다.
    for name in ('05_warped', '06_result'):
        stale = output / (name + '.png')
        if name not in result.stages and stale.exists():
            stale.unlink()
