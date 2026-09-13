"""처리 단계별 OpenCV 윈도우와 기본 트랙바 제어 패널입니다."""
import os
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .constants import MAX_BLUR_SIZE, MAX_PIXEL_VALUE, SCAN_STAGES
from .io import save_scan
from .pipeline import Parameters, Scan, scan

WINDOW = 'Controls'
TRACKBARS = ('Blur kernel', 'Canny low', 'Canny high')
PANEL_WIDTH, PANEL_HEIGHT = 984, 144
STAGE_WIDTH, STAGE_HEIGHT = 320, 180
FONT_DIRS = (Path('/usr/share/fonts/truetype/dejavu'),
             Path('/usr/share/fonts/truetype/liberation2'),
             Path('/usr/share/fonts/TTF'))
BACKGROUND = (245, 244, 240)
INK = (55, 48, 40)
ACCENT = (154, 103, 20)


def _configure_fonts() -> None:
    # Some Linux OpenCV wheels set this to a bundled directory that is absent.
    configured = os.environ.get('QT_QPA_FONTDIR')
    if not configured or Path(configured).is_dir():
        return
    for directory in FONT_DIRS:
        if any(directory.glob('*.ttf')):
            os.environ['QT_QPA_FONTDIR'] = str(directory)
            return


def _text(canvas, text, position, scale=0.55, color=INK, thickness=1):
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def render_controls(params: Parameters, status: str) -> np.ndarray:
    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), BACKGROUND, np.uint8)
    values = (f'{params.blur} x {params.blur}', str(params.low), str(params.high))
    for index, (label, value) in enumerate(zip(TRACKBARS, values)):
        left = 24 + index * 328
        _text(canvas, label, (left, 29))
        _text(canvas, value, (left, 65), 0.8, ACCENT, 2)
    _text(canvas, 'S: save   Q / Esc: quit   |   Adjust the trackbars below', (24, 101), 0.48)
    _text(canvas, status, (24, 129), 0.48)
    return canvas


def read_controls(params: Parameters) -> Parameters:
    positions = [cv2.getTrackbarPos(name, WINDOW) for name in TRACKBARS]
    blur, low, high = positions
    blur = min(MAX_BLUR_SIZE, max(1, blur) | 1)
    high = max(low + 1, high)
    current = replace(params, blur=blur, low=low, high=high)
    # Display exactly what scan() will use, including corrected even/invalid values.
    for name, before, after in zip(TRACKBARS, positions, (blur, low, high)):
        if before != after:
            cv2.setTrackbarPos(name, WINDOW, after)
    return current


def _show_stages(result: Scan) -> None:
    blank = np.full((STAGE_HEIGHT, STAGE_WIDTH, 3), BACKGROUND, np.uint8)
    _text(blank, 'No document found', (35, 94))
    for name in SCAN_STAGES:
        cv2.imshow(name, result.stages.get(name, blank))


def _arrange_windows() -> None:
    for index, name in enumerate(SCAN_STAGES):
        row, column = divmod(index, 3)
        cv2.resizeWindow(name, STAGE_WIDTH, STAGE_HEIGHT)
        cv2.moveWindow(name, 12 + column * 332, 12 + row * 216)
    cv2.resizeWindow(WINDOW, PANEL_WIDTH, PANEL_HEIGHT + 110)
    cv2.moveWindow(WINDOW, 12, 444)


def _windows_closed(names) -> bool:
    try:
        return any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in names)
    except cv2.error:
        return True  # A backend may discard a closed window immediately.


def interactive(params: Parameters, output: Path, image: np.ndarray) -> None:
    if sys.platform.startswith('linux') and not (
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    ):
        raise ValueError('No desktop display. Use --headless for image processing.')

    _configure_fonts()
    created = []
    previous = None
    displayed = None
    key = -1
    status = ''
    try:
        for name in (*SCAN_STAGES, WINDOW):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            created.append(name)
        cv2.imshow(WINDOW, render_controls(params, 'Preparing document...'))
        for name, value, maximum in zip(
            TRACKBARS, (params.blur, params.low, params.high),
            (MAX_BLUR_SIZE, MAX_PIXEL_VALUE - 1, MAX_PIXEL_VALUE),
        ):
            cv2.createTrackbar(name, WINDOW, value, maximum, lambda _: None)

        while True:
            current = read_controls(params)
            first_frame = previous is None
            if current != previous:
                result = scan(image, current)
                previous = current
                _show_stages(result)
                status = ('Document detected. Press S to save.' if result.corners is not None
                          else 'No document found; adjust parameters.')
            # Read the latest trackbar changes before handling a save event.
            if key in (ord('s'), ord('S')):
                if result.corners is None:
                    status = 'No document found; adjust parameters before saving.'
                else:
                    save_scan(result, output)
                    status = 'Saved. Adjust parameters or press Q to quit.'
                    print(f'Saved: {output.resolve()}')
            state = (current, status)
            if state != displayed:
                cv2.imshow(WINDOW, render_controls(current, status))
                displayed = state
            if first_frame:
                _arrange_windows()
            key = cv2.waitKey(30) & 0xff
            if key in (ord('q'), ord('Q'), 27) or _windows_closed(created):
                break
            if first_frame:
                # WSLg can overwrite initial placement while mapping the windows.
                _arrange_windows()
    finally:
        for name in reversed(created):
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass
