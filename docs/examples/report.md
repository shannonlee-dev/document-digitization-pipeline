# Evaluation results

Provenance: **synthetic**

Automatic geometric proxy: maximum matched corner distance / image diagonal <= 0.030, and warp produced.
This does not certify text quality or physical aspect ratio. Inspect saved warps before final submission.

| Preset | Condition | Total | Success | Failure | Rate |
|---|---|---:|---:|---:|---:|
| default | simple | 5 | 5 | 0 | 100% |
| default | shadow | 5 | 1 | 4 | 20% |
| default | tilted | 5 | 5 | 0 | 100% |
| default | complex | 5 | 2 | 3 | 40% |
| sensitive | simple | 5 | 5 | 0 | 100% |
| sensitive | shadow | 5 | 1 | 4 | 20% |
| sensitive | tilted | 5 | 5 | 0 | 100% |
| sensitive | complex | 5 | 2 | 3 | 40% |
| strict | simple | 5 | 5 | 0 | 100% |
| strict | shadow | 5 | 1 | 4 | 20% |
| strict | tilted | 5 | 4 | 1 | 80% |
| strict | complex | 5 | 2 | 3 | 40% |

## Per-image parameter comparison

| Image | Condition | default | sensitive | strict |
|---|---|---|---|---|
| simple_1 | simple | PASS (0.0000) | PASS (0.0000) | PASS (0.0010) |
| simple_2 | simple | PASS (0.0000) | PASS (0.0000) | PASS (0.0010) |
| simple_3 | simple | PASS (0.0000) | PASS (0.0000) | PASS (0.0000) |
| simple_4 | simple | PASS (0.0000) | PASS (0.0000) | PASS (0.0010) |
| simple_5 | simple | PASS (0.0000) | PASS (0.0000) | PASS (0.0010) |
| shadow_1 | shadow | PASS (0.0000) | PASS (0.0000) | PASS (0.0010) |
| shadow_2 | shadow | FAIL (not detected) | FAIL (0.1991) | FAIL (0.1991) |
| shadow_3 | shadow | FAIL (not detected) | FAIL (0.2002) | FAIL (0.2001) |
| shadow_4 | shadow | FAIL (not detected) | FAIL (0.2002) | FAIL (not detected) |
| shadow_5 | shadow | FAIL (0.2002) | FAIL (0.2002) | FAIL (not detected) |
| tilted_1 | tilted | PASS (0.0018) | PASS (0.0018) | PASS (0.0018) |
| tilted_2 | tilted | PASS (0.0006) | PASS (0.0006) | PASS (0.0019) |
| tilted_3 | tilted | PASS (0.0009) | PASS (0.0009) | PASS (0.0009) |
| tilted_4 | tilted | PASS (0.0011) | PASS (0.0006) | PASS (0.0011) |
| tilted_5 | tilted | PASS (0.0010) | PASS (0.0010) | FAIL (not detected) |
| complex_1 | complex | PASS (0.0000) | PASS (0.0000) | PASS (0.0010) |
| complex_2 | complex | PASS (0.0000) | PASS (0.0000) | PASS (0.0010) |
| complex_3 | complex | FAIL (0.2050) | FAIL (0.2057) | FAIL (0.2054) |
| complex_4 | complex | FAIL (0.2050) | FAIL (0.2050) | FAIL (0.2050) |
| complex_5 | complex | FAIL (0.2057) | FAIL (0.2057) | FAIL (0.2057) |
