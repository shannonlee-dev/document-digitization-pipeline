"""순수 이미지 처리 단계입니다. 화면과 파일 처리는 별도 모듈에 둡니다."""
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    DOCUMENT_CORNER_COUNT,
    MAX_BLUR_SIZE,
    MAX_PIXEL_VALUE,
    MIN_IMAGE_SIZE,
)

MAX_EPSILON = 0.2
MIN_BLOCK_SIZE = 3
MIN_WARP_SIZE = 2
CORNER_CROSS_TOLERANCE = 1e-3
CONTOUR_COLOR = (0, 0, 255)
CONTOUR_THICKNESS = 3
CORNER_LABEL_COLOR = (20, 255, 57)
CORNER_LABEL_SCALE = 1.7
CORNER_LABEL_THICKNESS = 2

@dataclass(frozen=True)
class Parameters:
    blur: int = 5
    low: int = 50
    high: int = 150
    min_area: float = 0.08
    epsilon: float = 0.02
    block_size: int = 31
    threshold_c: float = 10

    def __post_init__(self) -> None:
        if self.blur < 1 or self.blur > MAX_BLUR_SIZE or self.blur % 2 == 0:
            raise ValueError(f'blur must be odd and between 1 and {MAX_BLUR_SIZE}')
        if not 0 <= self.low < self.high <= MAX_PIXEL_VALUE:
            raise ValueError(f'Canny requires 0 <= low < high <= {MAX_PIXEL_VALUE}')
        if not 0 < self.min_area < 1 or not 0 < self.epsilon < MAX_EPSILON:
            raise ValueError(f'min_area must be in (0, 1), epsilon in (0, {MAX_EPSILON})')
        if self.block_size < MIN_BLOCK_SIZE or self.block_size % 2 == 0:
            raise ValueError(f'block_size must be odd and >= {MIN_BLOCK_SIZE}')


@dataclass
class Scan:
    stages: Dict[str, np.ndarray]
    corners: Optional[NDArray[np.float32]]


def scan(image: Optional[np.ndarray], params: Optional[Parameters] = None) -> Scan:
    params = params or Parameters()
    if (
        image is None
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < MIN_IMAGE_SIZE
    ):
        raise ValueError(
            f'Expected a uint8 BGR image of shape (H, W, 3), '
            f'at least {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}'
        )

    prepared = _preprocess(image, params) # gray + blur
    edges = cv2.Canny(prepared, params.low, params.high) #canny edge

    overlay = image.copy()
    stages = {
        '01_original': image,
        '02_preprocessed': prepared,
        '03_edges': edges,
        '04_contours': overlay,
    }

    corners = _detect_document(edges, params) # 4개 모서리 좌표

    if corners is not None:
        cv2.polylines(
            overlay, [corners.astype(np.int32)], True,
            CONTOUR_COLOR, CONTOUR_THICKNESS,
        )
        for index, point in enumerate(corners.astype(int)):
            cv2.putText(
                overlay, str(index), tuple(point),
                cv2.FONT_HERSHEY_SIMPLEX, CORNER_LABEL_SCALE,
                CORNER_LABEL_COLOR, CORNER_LABEL_THICKNESS,
            )

        warped = _warp_document(image, corners)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        stages['05_warped'] = warped
        stages['06_result'] = cv2.adaptiveThreshold(
            gray, MAX_PIXEL_VALUE,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            params.block_size, params.threshold_c,
        )

    return Scan(stages, corners)


def _order_corners(points: ArrayLike) -> NDArray[np.float32]:
    """중심 기준으로 정렬한 뒤 좌상단 점부터 시작하도록 회전합니다.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if points.shape != (DOCUMENT_CORNER_COUNT, 2) or not np.isfinite(points).all():
        raise ValueError('Exactly four finite corners are required')
    if len(np.unique(points, axis=0)) != DOCUMENT_CORNER_COUNT:
        raise ValueError('Corners must be distinct')

    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = min(
        range(DOCUMENT_CORNER_COUNT),
        key=lambda i: (
            float(ordered[i].sum()),
            float(ordered[i, 1]),
            float(ordered[i, 0]),
        ),
    )
    ordered = np.roll(ordered, -start, axis=0)

    edges = np.roll(ordered, -1, axis=0) - ordered

    # 다각형을 이루는 모든 벡터(x, y)에 대해, y벡터와 x벡터의 90도
    # 회전 벡터의 내적 부호가 항상 같으면(전부 0보다 크거나 전부 0보다 작으면) convex다.
    cross = (
        edges[:, 0] * np.roll(edges[:, 1], -1)
        - edges[:, 1] * np.roll(edges[:, 0], -1)
    )
    if np.any(cross <= CORNER_CROSS_TOLERANCE):
        raise ValueError('Corners must form a nondegenerate convex quadrilateral')
    return ordered


def _preprocess(image: np.ndarray, params: Parameters) -> np.ndarray:
    # OpenCV 색상 배열은 (높이, 너비, 채널) 형태의 BGR 순서입니다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (params.blur, params.blur), 0)


def _detect_document(
    edges: np.ndarray,
    params: Parameters,
) -> Optional[NDArray[np.float32]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_limit = edges.shape[0] * edges.shape[1] * params.min_area

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < area_limit:
            break

        epsilon = params.epsilon * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )
        if len(polygon) == DOCUMENT_CORNER_COUNT and cv2.isContourConvex(polygon):
            try:
                return _order_corners(polygon.reshape(DOCUMENT_CORNER_COUNT, 2))
            except ValueError:
                continue
    return None


def _warp_document(image: np.ndarray, corners: ArrayLike) -> np.ndarray:
    tl, tr, br, bl = _order_corners(corners)
    longest_horizontal_edge = max(
        MIN_WARP_SIZE,
        int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))),
    )
    longest_vertical_edge = max(
        MIN_WARP_SIZE,
        int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))),
    )

    destination = np.float32([
        [0, 0],
        [longest_horizontal_edge - 1, 0],
        [longest_horizontal_edge - 1, longest_vertical_edge - 1],
        [0, longest_vertical_edge - 1],
    ])
    transform = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), destination)
    return cv2.warpPerspective(image, transform, (longest_horizontal_edge, longest_vertical_edge))
