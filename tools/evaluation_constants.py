"""평가 도구에서 공유하는 상수를 정의합니다."""
from pathlib import Path

from scanner.pipeline import Parameters

GROUPS = ('simple', 'shadow', 'tilted', 'complex')
IMAGES_PER_GROUP = 5
REQUIRED_IMAGE_COUNT = len(GROUPS) * IMAGES_PER_GROUP

DEFAULT_OUTPUT = Path('outputs/evaluation')
DEFAULT_MANIFEST = Path('tools/manifest.json')
RESULTS_FILENAME = 'results.csv'

SUCCESS_PENDING = ''
SUCCESS_TRUE = 'true'
SUCCESS_FALSE = 'false'
VALID_SUCCESS_VALUES = (SUCCESS_PENDING, SUCCESS_TRUE, SUCCESS_FALSE)

PRESETS = {
    'default': Parameters(),
    'sensitive': Parameters(blur=3, low=10, high=40),
    'strict': Parameters(blur=9, low=100, high=220),
}

BRIGHTNESS_PERCENTILES = (5, 95)

HISTOGRAM_SHAPE = (400, 760, 3)
HISTOGRAM_COLORS = ((50, 130, 20), (180, 80, 0), (0, 90, 220), (150, 0, 140))
HISTOGRAM_LOG_SCALE = 10000
HISTOGRAM_X_ORIGIN = 40
HISTOGRAM_X_SCALE = 2.6
HISTOGRAM_Y_BASELINE = 340
HISTOGRAM_HEIGHT = 260
HISTOGRAM_LINE_THICKNESS = 2
HISTOGRAM_LEGEND_POSITION = (45, 30)
HISTOGRAM_LEGEND_SPACING = 175
HISTOGRAM_CAPTION_POSITION = (40, 380)

PLOT_TEXT_SCALE = 0.5
PLOT_TEXT_COLOR = (0, 0, 0)
PLOT_TEXT_THICKNESS = 1
