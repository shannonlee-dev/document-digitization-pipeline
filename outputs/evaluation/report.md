# 평가 결과

문서의 네 꼭짓점을 정확히 검출하고 문서를 반듯하게 보정한 경우 성공으로 판정합니다.

| 설정 | 조건 | 전체 | 성공 | 실패 | 미판정 | 성공률 |
|---|---|---:|---:|---:|---:|---:|
| default | complex | 5 | 3 | 2 | 0 | 60% |
| default | shadow | 5 | 5 | 0 | 0 | 100% |
| default | simple | 5 | 5 | 0 | 0 | 100% |
| default | tilted | 5 | 4 | 1 | 0 | 80% |
| default | ALL | 20 | 17 | 3 | 0 | 85% |
| sensitive | complex | 5 | 4 | 1 | 0 | 80% |
| sensitive | shadow | 5 | 5 | 0 | 0 | 100% |
| sensitive | simple | 5 | 5 | 0 | 0 | 100% |
| sensitive | tilted | 5 | 3 | 2 | 0 | 60% |
| sensitive | ALL | 20 | 17 | 3 | 0 | 85% |
| strict | complex | 5 | 3 | 2 | 0 | 60% |
| strict | shadow | 5 | 3 | 2 | 0 | 60% |
| strict | simple | 5 | 5 | 0 | 0 | 100% |
| strict | tilted | 5 | 2 | 3 | 0 | 40% |
| strict | ALL | 20 | 13 | 7 | 0 | 65% |
