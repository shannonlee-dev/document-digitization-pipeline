"""평가 이미지의 밝기 통계를 계산하고 히스토그램을 그립니다."""
from typing import Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np

from scanner.constants import GRAYSCALE_LEVELS, MAX_PIXEL_VALUE
from tools.evaluation_constants import (
    BRIGHTNESS_PERCENTILES,
    HISTOGRAM_CAPTION_POSITION,
    HISTOGRAM_COLORS,
    HISTOGRAM_HEIGHT,
    HISTOGRAM_LEGEND_POSITION,
    HISTOGRAM_LEGEND_SPACING,
    HISTOGRAM_LINE_THICKNESS,
    HISTOGRAM_LOG_SCALE,
    HISTOGRAM_SHAPE,
    HISTOGRAM_X_ORIGIN,
    HISTOGRAM_X_SCALE,
    HISTOGRAM_Y_BASELINE,
    PLOT_TEXT_COLOR,
    PLOT_TEXT_SCALE,
    PLOT_TEXT_THICKNESS,
)


def analyze_image(image: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """BGR 이미지의 밝기 통계와 정규화된 히스토그램을 반환합니다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = np.bincount(gray.ravel(), minlength=GRAYSCALE_LEVELS) / gray.size
    statistics = {
        'mean': float(gray.mean()),
        'std': float(gray.std()),
        'p05': float(np.percentile(gray, BRIGHTNESS_PERCENTILES[0])),
        'p95': float(np.percentile(gray, BRIGHTNESS_PERCENTILES[1])),
    }
    return statistics, histogram


def histogram_plot(histograms: Mapping[str, Sequence[np.ndarray]]) -> np.ndarray:
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
