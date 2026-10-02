"""Verify headless scanning through installed package imports."""

import cv2
import numpy as np
import pytest

from document_digitization.cli import main
from document_digitization.io import save_image


@pytest.fixture
def document_image(tmp_path):
    image = np.full((400, 500, 3), 30, dtype=np.uint8)
    corners = np.array([[110, 40], [370, 65], [395, 350], [90, 330]])
    cv2.fillConvexPoly(image, corners, (235, 235, 235))
    path = tmp_path / "문서.png"
    save_image(path, image)
    return path


@pytest.mark.smoke
def test_headless_scan_writes_binary_result(document_image, tmp_path):
    output = tmp_path / "scan"
    assert main(["--image", str(document_image), "--headless", "--output", str(output)]) == 0
    result = cv2.imread(str(output / "06_result.png"), cv2.IMREAD_GRAYSCALE)
    assert result is not None
    assert result.shape == (291, 306)
    assert set(np.unique(result)).issubset({0, 255})
