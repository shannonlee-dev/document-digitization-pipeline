# Document Digitization Pipeline

## 프로젝트 소개

OpenCV와 NumPy로 촬영한 문서를 검출하고, 원근을 보정한 뒤 이진화하는 Python 문서 스캐너입니다. 이미지 경로 입력과 웹캠을 지원하며, 데스크톱에서는 처리 단계와 실시간 파라미터 조절 화면을 제공합니다.

사용자 요청에 따라 생성한 **합성 이미지 20장**과 실측 분석 보고서를 포함합니다. 직접 촬영하거나 LMS에서 받은 사진은 아닙니다. 기본 설정의 합성 데이터 검출 성공률은 **65%(13/20)**이며 실제 사진 성능을 뜻하지 않습니다.

## 핵심 특징

- Grayscale → Gaussian blur → Canny → 윤곽선 → 네 꼭짓점 → 투시 변환 → 적응형 이진화
- 블러·Canny 트랙바, 단계별 화면, PNG 저장, 웹캠 캡처
- GUI 없는 환경용 `--headless`, 파일·파라미터 오류 처리
- 조건별 평가, 정답 꼭짓점 오차, 3개 설정 비교, 이미지별·조건별 밝기 히스토그램
- 실행 가능한 합성 데이터 생성기, 회귀 테스트, 실패 분석 보고서

## 아키텍처

```text
main.py                  CLI 진입점
scanner/pipeline.py      설정, 입출력, 전처리, 검출, 좌표 정렬, 변환, 이진화
scanner/cli.py           이미지/웹캠 실행, 트랙바와 화면, 저장 키 처리
tools/                  아래 두 명령을 python -m tools.… 형태로 실행
  evaluate.py            개인 이미지 20장 평가, CSV/JSON, 히스토그램과 단계별 이미지 출력
tests/test_scanner.py   기하, 파일 처리, CLI, GUI 제어 흐름 테스트
docs/analysis.md        실측 분석과 한계
  data/personal/          사용자가 촬영한 문서 이미지 20장
```

파이프라인 함수는 GUI 없이 사용할 수 있습니다. 환경별 설정은 불변 `Parameters`로 분리해 CLI와 평가에서 같은 값을 재사용합니다. 새로운 배경 대응은 `detect_document`에서 후보 선택 규칙을 수정하고, 같은 manifest에 대해 전후 결과를 비교하면 됩니다. 별도 플러그인 계층은 없습니다.

## 설치와 실행

Python 3.8 이상, OpenCV 4.5 이상, NumPy 1.20 이상이 필요합니다. 검증 환경은 Python 3.12.3 / OpenCV 5.0.0 / NumPy 2.5.3입니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --image "data/personal/image_cv (1).png"
```

Windows에서는 `python -m venv .venv`와 `.venv\Scripts\activate`를 사용합니다.

GUI 없는 서버·Colab에서는 다음 명령으로 중간 단계와 최종 이미지를 저장합니다.

```bash
python main.py --image "data/personal/image_cv (1).png" --headless --output outputs/scan
```

`--output`은 출력 디렉터리입니다. 두 번째 예시는 그림자 오검출을 재현하며, 사각형 검출만으로 실제 문서가 맞다는 보장은 없습니다. CLI는 검출 후보 유무를 판단하고, 정답과의 비교는 평가 도구가 담당합니다.

```bash
python main.py --webcam 0 --output outputs/camera
```

카메라 번호는 환경에 따라 바꿉니다. OS 카메라 권한이 필요합니다. `s`로 현재 결과를 저장하고 `q` 또는 Esc로 종료합니다. 웹캠은 `capture_000`, `capture_001` 순서로 저장하며 새 실행에서 번호가 다시 시작하므로 세션별 출력 경로를 사용하세요. 종료·오류 시 카메라 자원과 화면을 정리합니다.

### 화면과 조작

| 창 이름 | 내용 |
|---|---|
| `01_original` | BGR 원본 |
| `02_preprocessed` | Grayscale + Gaussian blur |
| `03_edges` | Canny 엣지 |
| `04_contours` | 검출 윤곽선과 꼭짓점 번호 0–3 |
| `05_warped` | 정면으로 보정한 컬러 이미지 |
| `06_result` | 적응형 이진화 결과 |
| `Controls` | Blur radius, Canny low, Canny high |

블러 커널은 `2 × radius + 1`입니다. 실제 high는 `max(low + 1, high 트랙바 값)`으로 정규화합니다. 이벤트 루프가 약 30ms 간격으로 값을 읽고 변경 시 전체 파이프라인을 다시 실행합니다. 웹캠은 프레임마다 처리하므로 실제 처리 속도는 해상도와 장비에 따라 달라집니다. 검출 실패 시 변환 창에 `No document found`를 표시합니다.

GUI에서는 `s`, headless에서는 실행 즉시 `01_original.png`부터 `06_result.png`까지 저장합니다. 같은 출력 경로는 덮어씁니다. 검출 실패 시 진단 단계 1–4만 남기고 이전 변환 결과 5–6을 제거합니다. `save_image(path, image)`는 PNG/JPEG를 지원하고 인코딩과 파일 저장을 확인합니다. 오류 시 메시지와 종료 코드 2, 성공 시 0을 반환합니다.

## 핵심 함수와 원리

| 함수 | 입력 → 출력 / 책임 |
|---|---|
| `read_image` | 파일 경로 → uint8 BGR 배열; 파일·디코딩 검사 |
| `preprocess` | BGR, 설정 → 흐림 처리된 회색조 배열 |
| `detect_document` | 엣지, 설정 → 정렬된 `(4,2)` 좌표 또는 None |
| `order_corners` | 순서 없는 네 점 → 좌상·우상·우하·좌하 |
| `warp_document` | BGR, 네 점 → 정면 보정 BGR |
| `scan` | BGR, 설정 → 단계별 배열과 좌표를 가진 `Scan` |
| `save_scan` | `Scan`, 디렉터리 → 단계별 PNG |

이미지는 `(Height, Width, Channel)` 배열이며 `image[y, x]`는 한 픽셀, `image[:, :, 0]`은 파란 채널입니다. NumPy로 `image[100:200, 50:150]` 영역을 자르거나 `image.mean(axis=(0, 1))`으로 채널별 평균을 구할 수 있습니다. OpenCV는 RGB가 아닌 **BGR** 순서를 씁니다.

가우시안 필터의 한 출력 픽셀은 주변 픽셀과 커널 가중치의 곱을 합한 값, 즉 국소 배열의 내적입니다. 잡음을 줄이지만 커널이 커지면 약한 문서 경계도 흐려집니다. Canny의 high 이상은 강한 엣지, low와 high 사이는 강한 엣지에 연결될 때 유지되는 약한 엣지입니다. low 미만은 제거합니다. 보고서의 `tilted_5`에서 강한 블러·높은 임계값 조합은 검출을 잃습니다.

닫힌 윤곽선 중 전체 이미지 면적의 8% 이상이고 `approxPolyDP` 결과가 볼록한 사각형인 가장 큰 후보를 선택합니다. 근사 오차는 둘레의 2%입니다. 이 규칙은 문서의 의미를 이해하지 못하므로 액자·책상 사각형을 잘못 고를 수 있습니다.

좌표 정렬은 중심을 구하고 각 점의 `atan2(y-cy, x-cx)` 순서로 정렬한 뒤 `x+y`가 최소인 점을 시작점으로 회전합니다. 이미지의 y축은 아래쪽이므로 좌상→우상→우하→좌하 순서가 됩니다. 동률은 y, x 순으로 결정하며 중복점·오목·퇴화 도형을 거부합니다. 독립적인 합/차 최솟값 방식에서 마름모 꼭짓점이 중복되는 문제를 피합니다. 대칭 도형의 의미상 위쪽이나 글씨 방향까지 판별하지는 않습니다.

`getPerspectiveTransform`은 네 대응점으로 3×3 호모그래피 H를 구합니다. `s[x', y', 1]ᵀ = H[x, y, 1]ᵀ`에서 마지막 좌표로 나누어 투영 좌표를 얻습니다. `warpPerspective`는 그 변환으로 픽셀을 재표본화합니다. 목표 폭·높이는 각각 마주 보는 두 변 길이 중 큰 값으로 추정합니다. 평면의 원근은 보정하지만 실제 종이 종횡비 복원, 구겨진 곡면 복원, 자동 글씨 방향 회전은 보장하지 않습니다.

이진화는 기본 31×31 주변의 가우시안 가중 밝기에서 C=10을 뺀 지역 임계값을 적용합니다. 완만한 조명 변화에는 도움이 되지만 잘못 검출된 영역이나 소실된 글자를 복구하지는 못합니다.

## 데이터와 평가 재현

```bash
python -m tools.evaluate data/personal --output outputs/evaluation
python -m unittest discover -s tests -v
```

`data/personal` 디렉터리의 PNG/JPEG 20장 × 설정 3개 = 60회 실행합니다. 개인 이미지에는 정답 꼭짓점이 없으므로 성공은 문서 사각형 검출과 변환 결과 생성 여부로 판정합니다. 평가 폴더에는 `results.csv`, `report.md`, `parameters.json`, `eda.json`, `histograms.png`와 이미지별 `01`–`20` 디렉터리가 생성됩니다.

manifest를 직접 사용할 때 `provenance`는 `personal`로 지정할 수 있습니다. `python -m tools.evaluate data/personal`처럼 이미지 디렉터리를 넘기면 PNG/JPEG 20장을 자동으로 읽습니다. 개인 이미지에는 정답 `corners`가 없어도 되며, 이 경우 검출·변환 성공 여부만 기록합니다.

성공 판정은 **네 점의 최대 대응 오차 / 이미지 대각선 ≤ 0.03이고 변환 결과가 존재**하는 기하학적 대리 기준입니다. 꼭짓점만 근사하게 맞아도 흐린 글씨·비율 왜곡은 남을 수 있으므로 최종 제출에서는 변환 이미지 육안 검토가 필요합니다. 탐지 자체와 정답 검출을 CSV의 `detected`, `success`로 구분합니다.

상세 분석은 [분석 보고서](docs/analysis.md)를 참고하세요. 데스크톱 창 표시·물리 카메라 동작은 이 서버에서 확인하지 못했으며 GUI 제어 흐름과 자원 해제는 모의 테스트로 검증했습니다.
