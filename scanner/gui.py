"""OpenCV 캔버스 하나에서 처리 단계와 파라미터를 표시합니다."""
import os
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .constants import MAX_BLUR_SIZE, MAX_PIXEL_VALUE, SCAN_STAGES
from .io import save_scan
from .pipeline import Parameters, Scan, scan

WINDOW = 'Document scanner'
WIDTH, HEIGHT = 1100, 800
SLIDER_LEFT, SLIDER_RIGHT = 300, 1048
SLIDER_ROWS = (568, 642, 716)
SLIDER_LIMITS = ((0, MAX_BLUR_SIZE // 2), (0, MAX_PIXEL_VALUE - 1), (1, MAX_PIXEL_VALUE))
LABELS = ('Gaussian blur', 'Canny low', 'Canny high')
STAGE_LABELS = ('01 Original', '02 Preprocessed', '03 Edges',
                '04 Contours', '05 Warped', '06 Result')
BACKGROUND = (245, 244, 240)
INK = (55, 48, 40)
MUTED = (115, 105, 95)
ACCENT = (154, 103, 20)
BORDER = (212, 206, 195)


def _text(canvas, text, position, scale=0.5, color=INK, thickness=1):
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


class Controls:
    """그리기와 이벤트가 동일한 좌표 및 실제 적용 파라미터를 공유합니다."""

    def __init__(self, params: Parameters):
        self.params = params
        self.selected = 0
        self.dragging = None

    def _set(self, index: int, value: int) -> None:
        if index == 0:
            radius = max(0, min(MAX_BLUR_SIZE // 2, value))
            self.params = replace(self.params, blur=radius * 2 + 1)
        elif index == 1:
            low = max(0, min(MAX_PIXEL_VALUE - 1, value))
            self.params = replace(self.params, low=low, high=max(low + 1, self.params.high))
        else:
            high = max(1, min(MAX_PIXEL_VALUE, value))
            self.params = replace(self.params, low=min(self.params.low, high - 1), high=high)

    def values(self):
        return (self.params.blur // 2, self.params.low, self.params.high)

    def mouse(self, event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = next((i for i, row in enumerate(SLIDER_ROWS)
                                  if abs(y - row) <= 20
                                  and SLIDER_LEFT - 12 <= x <= SLIDER_RIGHT + 12), None)
            if self.dragging is not None:
                self.selected = self.dragging
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None
            return
        elif event != cv2.EVENT_MOUSEMOVE:
            return
        elif not flags & cv2.EVENT_FLAG_LBUTTON:
            self.dragging = None
            return
        if self.dragging is not None:
            minimum, maximum = SLIDER_LIMITS[self.dragging]
            fraction = max(0.0, min(1.0, (x - SLIDER_LEFT) / (SLIDER_RIGHT - SLIDER_LEFT)))
            self._set(self.dragging, round(minimum + fraction * (maximum - minimum)))

    def key(self, key: int) -> None:
        if key == 9:
            self.selected = (self.selected + 1) % 3
        elif key in (ord('+'), ord('='), ord('-'), ord('_')):
            step = 1 if key in (ord('+'), ord('=')) else -1
            self._set(self.selected, self.values()[self.selected] + step)


def render(result: Scan, controls: Controls, status: str) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
    _text(canvas, 'DOCUMENT SCANNER', (24, 32), 0.7, thickness=2)
    _text(canvas, 'S  Save    Q / Esc  Quit', (805, 31))
    for index, (name, label) in enumerate(zip(SCAN_STAGES, STAGE_LABELS)):
        row, column = divmod(index, 3)
        left, top = 24 + column * 358, 52 + row * 218
        _text(canvas, label, (left, top + 16), color=MUTED)
        x, y, width, height = left, top + 27, 336, 176
        cv2.rectangle(canvas, (x, y), (x + width, y + height), BORDER, 1)
        source = result.stages.get(name)
        if source is None:
            _text(canvas, 'No document found', (x + 68, y + 92), color=MUTED)
            continue
        ratio = min(width / source.shape[1], height / source.shape[0])
        target = (max(1, round(source.shape[1] * ratio)),
                  max(1, round(source.shape[0] * ratio)))
        thumbnail = cv2.resize(source, target,
                               interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
        if thumbnail.ndim == 2:
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2BGR)
        x += (width - target[0]) // 2
        y += (height - target[1]) // 2
        canvas[y:y + target[1], x:x + target[0]] = thumbnail

    cv2.line(canvas, (24, 492), (1076, 492), BORDER, 1)
    _text(canvas, 'ADJUST', (24, 521), 0.55, thickness=2)
    _text(canvas, 'Drag a slider  |  Tab selects  |  +/- fine-tunes', (560, 521), color=MUTED)
    displayed = (f'{controls.params.blur} x {controls.params.blur}',
                 str(controls.params.low), str(controls.params.high))
    for index, (label, value, y, bounds) in enumerate(zip(LABELS, displayed, SLIDER_ROWS, SLIDER_LIMITS)):
        selected = index == controls.selected
        _text(canvas, label, (24, y - 5), 0.56, ACCENT if selected else INK,
              2 if selected else 1)
        _text(canvas, value, (213, y - 5), 0.5, ACCENT, 2)
        _text(canvas, 'Kernel size (odd)' if index == 0 else 'Edge threshold',
              (24, y + 17), 0.4, MUTED)
        minimum, maximum = bounds
        fraction = (controls.values()[index] - minimum) / (maximum - minimum)
        position = round(SLIDER_LEFT + fraction * (SLIDER_RIGHT - SLIDER_LEFT))
        cv2.line(canvas, (SLIDER_LEFT, y), (SLIDER_RIGHT, y), BORDER, 4)
        cv2.line(canvas, (SLIDER_LEFT, y), (position, y), ACCENT, 4)
        cv2.circle(canvas, (position, y), 9, ACCENT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (position, y), 4, BACKGROUND, -1, cv2.LINE_AA)
        _text(canvas, '1' if index == 0 else str(minimum), (SLIDER_LEFT, y + 25), 0.4, MUTED)
        _text(canvas, '51' if index == 0 else str(maximum), (SLIDER_RIGHT - 24, y + 25), 0.4, MUTED)
    cv2.line(canvas, (24, 757), (1076, 757), BORDER, 1)
    _text(canvas, status, (24, 784), 0.48)
    return canvas


def interactive(params: Parameters, output: Path, image: np.ndarray) -> None:
    if sys.platform.startswith('linux') and not (
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    ):
        raise ValueError('No desktop display. Use --headless for image processing.')

    controls = Controls(params)
    previous = None
    result = None
    status = ''
    key = -1
    displayed = None
    created = False
    try:
        # GUI_NORMAL removes Qt's image toolbar; the entire client area is our canvas.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        created = True
        cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)
        cv2.setMouseCallback(WINDOW, controls.mouse)
        while True:
            if controls.params != previous:
                result = scan(image, controls.params)
                previous = controls.params
                status = ('Document detected. Press S to save.' if result.corners is not None
                          else 'No document found; adjust parameters.')
            if key in (ord('s'), ord('S')):
                if result.corners is None:
                    status = 'No document found; adjust parameters before saving.'
                else:
                    save_scan(result, output)
                    status = 'Saved. Adjust parameters or press Q to quit.'
                    print(f'Saved: {output.resolve()}')
            state = (previous, controls.selected, status)
            if state != displayed:
                cv2.imshow(WINDOW, render(result, controls, status))
                displayed = state
            key = cv2.waitKey(30) & 0xff
            if key in (ord('q'), ord('Q'), 27):
                break
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break  # Some backends discard the window immediately on close.
            controls.key(key)
    finally:
        if created:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass  # The user may already have closed it.
