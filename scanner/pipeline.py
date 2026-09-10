"""Pure processing stages; GUI and filesystem concerns live in separate modules."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Parameters:
    blur: int = 5
    low: int = 50
    high: int = 150
    min_area: float = 0.08
    epsilon: float = 0.02
    block_size: int = 31
    threshold_c: float = 10

    def __post_init__(self):
        if self.blur < 1 or self.blur > 51 or self.blur % 2 == 0:
            raise ValueError('blur must be odd and between 1 and 51')
        if not 0 <= self.low < self.high <= 255:
            raise ValueError('Canny requires 0 <= low < high <= 255')
        if not 0 < self.min_area < 1 or not 0 < self.epsilon < 0.2:
            raise ValueError('min_area must be in (0, 1), epsilon in (0, 0.2)')
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError('block_size must be odd and >= 3')


@dataclass
class Scan:
    stages: dict
    corners: Optional[np.ndarray]


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Image does not exist: {}'.format(path))
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 3:
        raise ValueError('Not a readable image: {}'.format(path))
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


def order_corners(points):
    """Sort around centroid, then rotate to the upper-left (minimum x+y).

    Angular sorting avoids duplicate corners caused by independent sum/difference
    argmin rules for diamonds. In image coordinates y points down, so increasing
    atan2 produces TL, TR, BR, BL. Ties use y then x deterministically.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Exactly four finite corners are required')
    if len(np.unique(points, axis=0)) != 4:
        raise ValueError('Corners must be distinct')
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = min(range(4), key=lambda i: (float(ordered[i].sum()), float(ordered[i, 1]), float(ordered[i, 0])))
    ordered = np.roll(ordered, -start, axis=0)
    edges = np.roll(ordered, -1, axis=0) - ordered
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(cross <= 1e-3):
        raise ValueError('Corners must form a nondegenerate convex quadrilateral')
    return ordered


def preprocess(image, params):
    # OpenCV color arrays have shape (height, width, channels), in BGR order.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (params.blur, params.blur), 0)


def detect_document(edges, params):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_limit = edges.shape[0] * edges.shape[1] * params.min_area
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < area_limit:
            break
        polygon = cv2.approxPolyDP(contour, params.epsilon * cv2.arcLength(contour, True), True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            try:
                return order_corners(polygon.reshape(4, 2))
            except ValueError:
                continue
    return None


def warp_document(image, corners):
    tl, tr, br, bl = order_corners(corners)
    width = max(2, int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))))
    height = max(2, int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))))
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), destination)
    return cv2.warpPerspective(image, transform, (width, height))


def scan(image, params=None):
    params = params or Parameters()
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 3:
        raise ValueError('Expected a uint8 BGR image of shape (H, W, 3), at least 3x3')
    prepared = preprocess(image, params)
    edges = cv2.Canny(prepared, params.low, params.high)
    corners = detect_document(edges, params)
    overlay = image.copy()
    stages = {'01_original': image, '02_preprocessed': prepared, '03_edges': edges, '04_contours': overlay}
    if corners is not None:
        cv2.polylines(overlay, [corners.astype(np.int32)], True, (0, 0, 255), 3)
        for index, point in enumerate(corners.astype(int)):
            cv2.putText(overlay, str(index), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
        warped = warp_document(image, corners)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        stages['05_warped'] = warped
        stages['06_result'] = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, params.block_size, params.threshold_c)
    return Scan(stages, corners)


def save_scan(result, output):
    output = Path(output)
    for name, image in result.stages.items():
        save_image(output / (name + '.png'), image)
    # A failed rerun must not leave a previous scan looking like its result.
    for name in ('05_warped', '06_result'):
        stale = output / (name + '.png')
        if name not in result.stages and stale.exists():
            stale.unlink()
