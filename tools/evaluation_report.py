"""수동 평가 CSV를 검증하고 판정 결과를 Markdown 보고서로 집계합니다."""
import csv
from pathlib import Path

from tools.evaluation_constants import (
    RESULTS_FILENAME,
    SUCCESS_FALSE,
    SUCCESS_TRUE,
    VALID_SUCCESS_VALUES,
)


def summarize(output: Path) -> None:
    """수동 판정 CSV를 집계하며 미판정은 성공·실패에 포함하지 않습니다."""
    with (output / RESULTS_FILENAME).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('results.csv is empty')
    for row in rows:
        row['success'] = row['success'].strip().lower()
        row['success'] = {'1': SUCCESS_TRUE, '0': SUCCESS_FALSE}.get(
            row['success'], row['success'],
        )
        if row['success'] not in VALID_SUCCESS_VALUES:
            raise ValueError('success에는 true/false, 1/0 또는 빈 값을 입력하세요')

    lines = [
        '# 평가 결과', '',
        '문서의 네 꼭짓점을 정확히 검출하고 문서를 반듯하게 보정한 경우 성공으로 판정합니다.',
        '',
        '| 설정 | 조건 | 전체 | 성공 | 실패 | 미판정 | 성공률 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for preset in dict.fromkeys(r['preset'] for r in rows):
        preset_rows = [r for r in rows if r['preset'] == preset]
        for group in sorted({r['condition'] for r in preset_rows}) + ['ALL']:
            subset = [r for r in preset_rows if group == 'ALL' or r['condition'] == group]
            successes = sum(r['success'] == SUCCESS_TRUE for r in subset)
            failures = sum(r['success'] == SUCCESS_FALSE for r in subset)
            pending = len(subset) - successes - failures
            rate = 'pending' if pending else f'{successes / len(subset):.0%}'
            lines.append(
                f'| {preset} | {group} | {len(subset)} | {successes} | '
                f'{failures} | {pending} | {rate} |'
            )
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
