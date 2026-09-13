import os
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scanner import gui
from scanner.pipeline import Parameters, scan


@contextmanager
def desktop(keys=(ord('q'),), positions=(5, 50, 150)):
    with ExitStack() as stack:
        stack.enter_context(patch.dict('os.environ', {'DISPLAY': ':test'}))
        mocks = {name: stack.enter_context(patch('scanner.gui.cv2.' + name)) for name in (
            'namedWindow', 'resizeWindow', 'moveWindow', 'imshow', 'createTrackbar',
            'getTrackbarPos', 'setTrackbarPos', 'waitKey', 'getWindowProperty', 'destroyWindow')}
        mocks['waitKey'].side_effect = keys
        mocks['getTrackbarPos'].side_effect = lambda name, window: positions[gui.TRACKBARS.index(name)]
        mocks['getWindowProperty'].return_value = 1
        yield mocks


class GuiTests(unittest.TestCase):
    def test_broken_qt_font_directory_uses_system_fonts(self):
        with tempfile.TemporaryDirectory() as directory:
            fonts = Path(directory)
            (fonts / 'test.ttf').touch()
            with patch.dict(os.environ, {'QT_QPA_FONTDIR': '/missing/fonts'}), patch.object(gui, 'FONT_DIRS', (fonts,)):
                gui._configure_fonts()
                self.assertEqual(os.environ['QT_QPA_FONTDIR'], directory)
            with patch.dict(os.environ, {'QT_QPA_FONTDIR': directory}), patch.object(gui, 'FONT_DIRS', ()):
                gui._configure_fonts()
                self.assertEqual(os.environ['QT_QPA_FONTDIR'], directory)

    def test_trackbar_values_are_valid_and_synced(self):
        with desktop(positions=(0, 254, 0)) as ui:
            current = gui.read_controls(Parameters(min_area=0.2))
            self.assertEqual((current.blur, current.low, current.high), (1, 254, 255))
            self.assertEqual(current.min_area, 0.2)
            self.assertEqual(ui['setTrackbarPos'].call_count, 2)
        with desktop(positions=(50, 50, 150)):
            self.assertEqual(gui.read_controls(Parameters()).blur, 51)

    def test_separate_stage_windows_native_trackbars_and_cleanup(self):
        with desktop() as ui:
            gui.interactive(Parameters(), Path('/tmp/unused'), np.zeros((100, 100, 3), np.uint8))
            names = {call.args[0] for call in ui['namedWindow'].call_args_list}
            self.assertEqual(names, set(gui.SCAN_STAGES) | {gui.WINDOW})
            self.assertEqual(ui['createTrackbar'].call_count, 3)
            self.assertEqual({call.args[0] for call in ui['destroyWindow'].call_args_list}, names)
            shown = {call.args[0]: call.args[1] for call in ui['imshow'].call_args_list}
            self.assertTrue(all(name in shown for name in gui.SCAN_STAGES))
            self.assertGreater(shown['05_warped'].max(), 0)

    def test_panel_shows_actual_values(self):
        with patch('scanner.gui.cv2.putText', wraps=cv2.putText) as text:
            frame = gui.render_controls(Parameters(), 'Ready')
        labels = ' '.join(call.args[1] for call in text.call_args_list)
        for expected in ('Blur kernel', '5 x 5', 'Canny low', '50', 'Canny high', '150', 'Ready'):
            self.assertIn(expected, labels)
        self.assertEqual(frame.shape, (gui.PANEL_HEIGHT, gui.PANEL_WIDTH, 3))

    def test_save_uses_latest_trackbar_change_without_reprocessing_idle(self):
        image = np.zeros((300, 400, 3), np.uint8)
        cv2.rectangle(image, (50, 40), (350, 260), (255, 255, 255), -1)
        positions = [5, 50, 150]
        with tempfile.TemporaryDirectory() as directory, desktop(positions=positions) as ui:
            def key(delay):
                positions[0] = 7
                return (ord('s'), -1, ord('q'))[ui['waitKey'].call_count - 1]
            ui['waitKey'].side_effect = key
            with patch('scanner.gui.scan', wraps=scan) as process:
                gui.interactive(Parameters(), Path(directory), image)
            self.assertEqual(process.call_count, 2)
            expected = scan(image, Parameters(blur=7)).stages['02_preprocessed']
            actual = cv2.imread(str(Path(directory) / '02_preprocessed.png'), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(actual, expected)

    def test_missing_document_not_saved_and_closed_window_exits(self):
        with desktop(keys=(ord('s'), ord('q'))), patch('scanner.gui.save_scan') as save:
            gui.interactive(Parameters(), Path('/tmp/unused'), np.zeros((100, 100, 3), np.uint8))
            save.assert_not_called()
        with desktop(keys=(-1,)) as ui:
            ui['getWindowProperty'].return_value = 0
            gui.interactive(Parameters(), Path('/tmp/unused'), np.zeros((100, 100, 3), np.uint8))
            self.assertEqual(ui['waitKey'].call_count, 1)

    def test_scan_failure_closes_created_windows(self):
        with desktop() as ui, patch('scanner.gui.scan', side_effect=ValueError('failed')):
            with self.assertRaises(ValueError):
                gui.interactive(Parameters(), Path('/tmp/unused'), np.zeros((100, 100, 3), np.uint8))
            self.assertEqual(ui['destroyWindow'].call_count, ui['namedWindow'].call_count)
