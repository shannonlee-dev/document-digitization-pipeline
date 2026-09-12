"""스캔 처리, 입출력, 화면 및 평가 도구가 공유하는 고정값입니다."""

SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg')
STAGE_IMAGE_SUFFIX = '.png'
MAX_PIXEL_VALUE = 255
GRAYSCALE_LEVELS = MAX_PIXEL_VALUE + 1
MIN_IMAGE_SIZE = 3
DOCUMENT_CORNER_COUNT = 4
MAX_BLUR_SIZE = 51

OUTPUT_STAGES = ('05_warped', '06_result')
SCAN_STAGES = (
    '01_original',
    '02_preprocessed',
    '03_edges',
    '04_contours',
    *OUTPUT_STAGES,
)
