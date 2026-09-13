"""수동 평가 CSV를 검증하고 판정 결과를 Markdown 보고서로 집계합니다."""
import csv
from pathlib import Path


def summarize(output: Path) -> None:
    """수동 판정 CSV를 집계하며 미판정은 성공·실패에 포함하지 않습니다."""
    with (output / 'results.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('results.csv is empty')
    for row in rows:
        row['success'] = row['success'].strip().lower()
        if row['success'] not in ('', 'true', 'false'):
            raise ValueError('success must be true, false, or blank')

    lines = [
        '# Evaluation results', '',
        'Manual review: inspect 04_contours.png, 05_warped.png and 06_result.png.',
        'PASS requires four correct document corners and a straightened document.',
        'Fill success with true/false in results.csv; leave unreviewed rows blank.',
        'Record failure reasons and parameter adjustment results in review_notes.',
        'Analyze at least three failures. Rates appear only after every row in a group is reviewed.',
        '',
        '| Preset | Condition | Total | Success | Failure | Pending | Rate |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for preset in dict.fromkeys(r['preset'] for r in rows):
        preset_rows = [r for r in rows if r['preset'] == preset]
        for group in sorted({r['condition'] for r in preset_rows}) + ['ALL']:
            subset = [r for r in preset_rows if group == 'ALL' or r['condition'] == group]
            successes = sum(r['success'] == 'true' for r in subset)
            failures = sum(r['success'] == 'false' for r in subset)
            pending = len(subset) - successes - failures
            rate = 'pending' if pending else f'{successes / len(subset):.0%}'
            lines.append(
                f'| {preset} | {group} | {len(subset)} | {successes} | '
                f'{failures} | {pending} | {rate} |'
            )
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
