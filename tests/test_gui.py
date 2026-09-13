import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scanner import gui
from scanner.pipeline import Parameters, scan


class GuiTests(unittest.TestCase):
    def test_sliders_drag_clamp_and_preserve_parameters(self):
        panel = gui.Controls(Parameters(min_area=0.2))
        x0, x1 = gui.SLIDER_LEFT, gui.SLIDER_RIGHT
        panel.mouse(cv2.EVENT_LBUTTONDOWN, x1, gui.SLIDER_ROWS[0], 0, None)
        self.assertEqual(panel.params.blur, 51)
        panel.mouse(cv2.EVENT_MOUSEMOVE, x0 - 100, 0, cv2.EVENT_FLAG_LBUTTON, None)
        self.assertEqual(panel.params.blur, 1)
        panel.mouse(cv2.EVENT_LBUTTONUP, x0, 0, 0, None)
        panel.mouse(cv2.EVENT_MOUSEMOVE, x1, gui.SLIDER_ROWS[0], 0, None)
        self.assertEqual(panel.params.blur, 1)
        panel.mouse(cv2.EVENT_LBUTTONDOWN, x1, gui.SLIDER_ROWS[1], 0, None)
        self.assertEqual((panel.params.low, panel.params.high), (254, 255))
        panel.mouse(cv2.EVENT_LBUTTONDOWN, x0, gui.SLIDER_ROWS[2], 0, None)
        self.assertEqual((panel.params.low, panel.params.high), (0, 1))
        self.assertEqual(panel.params.min_area, 0.2)

    def test_keyboard_adjusts_selected_slider(self):
        panel = gui.Controls(Parameters())
        panel.key(ord('+'))
        self.assertEqual(panel.params.blur, 7)
        panel.key(9)
        panel.key(ord('-'))
        self.assertEqual(panel.params.low, 49)

    def test_render_contains_labels_values_and_missing_document_message(self):
        panel = gui.Controls(Parameters())
        result = scan(np.zeros((100, 200, 3), np.uint8))
        with patch('scanner.gui.cv2.putText', wraps=cv2.putText) as draw:
            frame = gui.render(result, panel, 'No document found; adjust parameters.')
        labels = [call.args[1] for call in draw.call_args_list]
        for label in ('Gaussian blur', '5 x 5', 'Canny low', '50', 'Canny high', '150', 'No document found'):
            self.assertIn(label, labels)
        self.assertEqual(frame.shape, (gui.HEIGHT, gui.WIDTH, 3))

    @patch.dict('os.environ', {'DISPLAY': ':test'})
    def test_reprocess_save_and_cleanup(self):
        image = np.zeros((300, 400, 3), np.uint8)
        cv2.rectangle(image, (50, 40), (350, 260), (255, 255, 255), -1)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('scanner.gui.cv2.namedWindow'),
            patch('scanner.gui.cv2.resizeWindow'),
            patch('scanner.gui.cv2.imshow'),
            patch('scanner.gui.cv2.setMouseCallback'),
            patch('scanner.gui.cv2.waitKey', side_effect=[ord('+'), ord('s'), ord('q')]),
            patch('scanner.gui.cv2.getWindowProperty', return_value=1),
            patch('scanner.gui.cv2.destroyWindow') as destroy,
            patch('scanner.gui.scan', wraps=scan) as process,
        ):
            gui.interactive(Parameters(), Path(directory), image)
            self.assertEqual(process.call_count, 2)
            self.assertEqual(process.call_args.args[1].blur, 7)
            self.assertTrue((Path(directory) / '06_result.png').exists())
            destroy.assert_called_once_with(gui.WINDOW)

    @patch.dict('os.environ', {'DISPLAY': ':test'})
    def test_window_close_and_scan_error_cleanup(self):
        for failure in (False, True):
            with (
                patch('scanner.gui.cv2.namedWindow'),
                patch('scanner.gui.cv2.resizeWindow'),
                patch('scanner.gui.cv2.imshow'),
                patch('scanner.gui.cv2.setMouseCallback'),
                patch('scanner.gui.cv2.waitKey', return_value=-1),
                patch('scanner.gui.cv2.getWindowProperty', return_value=0),
                patch('scanner.gui.cv2.destroyWindow') as destroy,
            ):
                image = np.zeros((100, 100, 3), np.uint8)
                if failure:
                    with patch('scanner.gui.scan', side_effect=ValueError('failed')):
                        with self.assertRaises(ValueError):
                            gui.interactive(Parameters(), Path('/tmp/unused'), image)
                else:
                    gui.interactive(Parameters(), Path('/tmp/unused'), image)
                destroy.assert_called_once()

    @patch.dict('os.environ', {'DISPLAY': ':test'})
    def test_save_uses_latest_mouse_change_and_rejects_missing_document(self):
        for detected in (True, False):
            image = np.zeros((300, 400, 3), np.uint8)
            if detected:
                cv2.rectangle(image, (50, 40), (350, 260), (255, 255, 255), -1)
            with (
                tempfile.TemporaryDirectory() as directory,
                patch('scanner.gui.cv2.namedWindow'),
                patch('scanner.gui.cv2.resizeWindow'),
                patch('scanner.gui.cv2.imshow'),
                patch('scanner.gui.cv2.setMouseCallback') as callback,
                patch('scanner.gui.cv2.getWindowProperty', return_value=1),
                patch('scanner.gui.cv2.destroyWindow'),
                patch('scanner.gui.save_scan', wraps=gui.save_scan) as save,
            ):
                def change_and_save(delay):
                    mouse = callback.call_args.args[1]
                    mouse(cv2.EVENT_LBUTTONDOWN,
                          gui.SLIDER_LEFT + round((gui.SLIDER_RIGHT - gui.SLIDER_LEFT) * 3 / 25),
                          gui.SLIDER_ROWS[0], 0, None)
                    return ord('s')

                with patch('scanner.gui.cv2.waitKey') as wait:
                    wait.side_effect = lambda delay: change_and_save(delay) if wait.call_count == 1 else ord('q')
                    gui.interactive(Parameters(), Path(directory), image)
                self.assertEqual(save.call_count, int(detected))
                if detected:
                    expected = scan(image, Parameters(blur=7))
                    np.testing.assert_array_equal(save.call_args.args[0].stages['02_preprocessed'],
                                                  expected.stages['02_preprocessed'])
