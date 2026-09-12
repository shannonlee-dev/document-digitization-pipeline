import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scanner.cli import _interactive, main
from scanner.io import read_image, save_image, save_scan
from scanner.pipeline import Parameters, _order_corners, scan, _warp_document
from tools.evaluate import _corner_error, _load_manifest


class ScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((400, 500, 3), 30, np.uint8)
        self.corners = np.float32([[110, 40], [370, 65], [395, 350], [90, 330]])
        cv2.fillConvexPoly(self.image, self.corners.astype(int), (235, 235, 235))
        cv2.putText(
            self.image, 'SCAN', (160, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2,
        )

    def test_every_corner_permutation(self) -> None:
        for permutation in itertools.permutations(self.corners):
            np.testing.assert_allclose(_order_corners(permutation), self.corners)

    def test_diamond_and_invalid_corners(self) -> None:
        diamond = [[100, 0], [200, 100], [100, 200], [0, 100]]
        self.assertEqual(len(np.unique(_order_corners(diamond), axis=0)), 4)
        for points in (
            [[0, 0]] * 4,
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [[0, 0], [10, 0], [1, 1], [0, 10]],
        ):
            with self.assertRaises(ValueError):
                _order_corners(points)

    def test_scan_geometry_and_binary(self) -> None:
        original = self.image.copy()
        result = scan(self.image)

        self.assertIsNotNone(result.corners)
        self.assertLess(_corner_error(result.corners, self.corners, self.image.shape), .01)
        self.assertEqual(len(result.stages), 6)
        self.assertTrue(set(np.unique(result.stages['06_result'])).issubset({0, 255}))
        np.testing.assert_array_equal(self.image, original)
        self.assertEqual(_warp_document(self.image, self.corners).shape[:2], (291, 306))

    def test_blank_and_invalid_image(self) -> None:
        self.assertIsNone(scan(np.zeros((100, 100, 3), np.uint8)).corners)
        with self.assertRaises(ValueError):
            scan(np.zeros((10, 10), np.uint8))

    def test_parameter_validation(self) -> None:
        for kwargs in (
            {'blur': 2},
            {'low': 200, 'high': 100},
            {'min_area': 0},
            {'block_size': 2},
        ):
            with self.assertRaises(ValueError):
                Parameters(**kwargs)

    def test_files_and_headless_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / '문서.png'
            save_image(path, self.image)
            np.testing.assert_array_equal(read_image(path), self.image)
            self.assertEqual(
                main([
                    '--image', str(path), '--headless',
                    '--output', str(root / 'result'),
                ]),
                0,
            )
            self.assertTrue((root / 'result/06_result.png').exists())

            save_scan(scan(np.zeros_like(self.image)), root / 'result')
            self.assertFalse((root / 'result/06_result.png').exists())

            self.assertEqual(main(['--image', str(root / 'missing.png'), '--headless']), 2)
            bad = root / 'bad.png'
            bad.write_text('not an image')
            with self.assertRaises(ValueError):
                read_image(bad)
            with self.assertRaises(ValueError):
                save_image(root / 'bad.gif', self.image)

    def test_wrong_rectangle_is_not_success(self) -> None:
        self.assertGreater(
            _corner_error(
                [[0, 0], [499, 0], [499, 399], [0, 399]],
                self.corners, self.image.shape,
            ),
            .03,
        )

    def test_manifest_requires_full_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text('{"provenance": "personal", "images": []}')
            with self.assertRaises(ValueError):
                _load_manifest(path)

    @patch.dict('os.environ', {'DISPLAY': ':test'})
    def test_trackbar_reprocess_and_save(self) -> None:
        calls = [0]

        def position(name: str, window: str) -> int:
            if name == 'Blur radius':
                calls[0] += 1
            return {
                'Blur radius': 2 if calls[0] == 1 else 3,
                'Canny low': 50,
                'Canny high': 150,
            }[name]

        with (
            tempfile.TemporaryDirectory() as directory,
            patch('scanner.cli.cv2.namedWindow'),
            patch('scanner.cli.cv2.createTrackbar') as tracks,
            patch('scanner.cli.cv2.getTrackbarPos', side_effect=position),
            patch('scanner.cli.cv2.waitKey', side_effect=[ord('s'), ord('q')]),
            patch('scanner.cli.cv2.getWindowProperty', return_value=1),
            patch('scanner.cli.cv2.destroyAllWindows'),
            patch('scanner.cli._show') as show,
        ):
            _interactive(Parameters(), Path(directory), image=self.image)
            self.assertEqual(tracks.call_count, 3)
            self.assertEqual(show.call_count, 2)
            self.assertTrue((Path(directory) / '06_result.png').exists())


if __name__ == '__main__':
    unittest.main()
