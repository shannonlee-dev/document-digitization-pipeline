# Document Digitization Pipeline

## 프로젝트 소개

OpenCV와 NumPy로 문서 사진을 검출하고 원근 보정·이진화하는 작은 Python 스캐너입니다. 이미지 파일과 웹캠을 지원하며, 학습 모델이나 OCR은 사용하지 않습니다.

## 핵심 특징

- 회색조 → Gaussian blur → Canny → 사각형 검출 → 투시 변환 → 적응형 이진화
- 단계별 미리보기, 블러·Canny 트랙바, PNG 저장
- GUI 없는 환경의 파일 처리와 이미지 20장에 대한 설정 비교

## 아키텍처

```text
main.py                 실행 진입점
scanner/
  pipeline.py           Parameters, Scan, 전처리·검출·좌표 정렬·변환
  io.py                 이미지 읽기와 단계별 저장
  cli.py                인자 처리, 화면·트랙바, 웹캠 제어
tools/
  evaluate.py           데이터 로딩, 평가, 히스토그램·보고서 생성
tests/
  test_scanner.py       기하·입출력·CLI·GUI 제어 흐름 검증
data/personal/         평가 이미지 20장
outputs/              스캔 결과와 평가 산출물
```

`scanner/cli.py`와 `tools/evaluate.py`가 같은 `scan(image, params)`를 호출합니다. 처리 과정은 `scanner/pipeline.py`를 위에서 아래로 읽으면 따라갈 수 있습니다. 설정과 결과는 두 dataclass로 표현하고, 파일 입출력은 `scanner/io.py`에 둡니다.

## 설치와 실행

Python 3.8 이상이 필요합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --image data/personal/simple_1.png
```

Windows에서는 `python -m venv .venv`와 `.venv\Scripts\activate`를 사용합니다.

```bash
# 화면 없이 처리하고 저장
python main.py --image data/personal/simple_1.png --headless --output outputs/scan

# 웹캠: 데스크톱 화면과 카메라 필요
python main.py --webcam 0 --output outputs/camera

# 전체 옵션과 기본값
python main.py --help
```

GUI에서 블러·Canny 트랙바로 설정을 조절하고, `s`로 저장하며 `q` 또는 Esc로 종료합니다. 웹캠은 프레임마다 처리하고 `capture_000`부터 저장합니다. 번호는 실행마다 초기화되므로 세션별 출력 경로를 사용하세요.

저장 단계는 `01_original`, `02_preprocessed`, `03_edges`, `04_contours`, `05_warped`, `06_result`이며 각각 PNG 파일로 생성됩니다. Headless는 즉시 저장합니다. 같은 경로는 덮어쓰며, 검출 실패 시 진단 단계 1–4만 남기고 이전 결과 5–6을 지웁니다. 성공 종료 코드는 0, 오류는 2입니다.

Python에서도 직접 사용할 수 있습니다.

```python
from scanner.io import read_image, save_scan
from scanner.pipeline import Parameters, scan

image = read_image('data/personal/simple_1.png')
result = scan(image, Parameters(blur=5, low=50, high=150))
save_scan(result, 'outputs/scan')
```

입력은 `uint8` BGR 배열 `(H, W, 3)`입니다. `Scan.stages`에는 처리 단계별 배열, `Scan.corners`에는 정렬된 네 꼭짓점 또는 검출 실패를 뜻하는 `None`이 담깁니다.

## 평가와 테스트

```bash
python -m tools.evaluate data/personal --output outputs/evaluation
python -m tools.evaluate data/personal --output outputs/evaluation --epsilon-sweep
python -m unittest discover -s tests -v
```

디렉터리 입력은 PNG/JPEG 20장을 파일명 순으로 읽습니다. 각 이미지를 `default`(blur=5, Canny=50/150), `sensitive`(3, 10/40), `strict`(9, 100/220)로 처리합니다. `--epsilon-sweep`은 사각형 근사 비율 0.01, 0.02, 0.04, 0.06도 비교합니다.

평가 폴더에는 `results.csv`, `parameters.json`, `eda.json`, `histograms.png`, `report.md`와 이미지별 단계·미리보기가 생성됩니다. 상세 결과는 생성된 `report.md`를 기준으로 확인합니다.

현재 포함된 개인 이미지 20장에는 정답 꼭짓점이 없으므로 **사각형 검출과 변환 결과 생성 여부**만 평가합니다. 포함된 데이터의 실행 결과는 다음과 같습니다.

| 설정 | 성공 | 실패 | 성공률 |
|---|---:|---:|---:|
| default | 18 | 2 | 90% |
| sensitive | 18 | 2 | 90% |
| strict | 14 | 6 | 70% |

정답 좌표로 평가하려면 디렉터리 대신 JSON manifest를 전달합니다. `provenance`는 `personal`, `synthetic`, `lms` 중 하나이며, `images`에는 `simple`, `shadow`, `tilted`, `complex` 조건별 5장씩 총 20장이 필요합니다. 각 항목은 고유한 `id`, manifest 기준 상대 `path`, `condition`, 촬영 조건 `notes`를 갖습니다. 선택 항목 `corners`는 이미지 안의 네 `[x, y]` 좌표입니다.

정답이 있는 이미지는 최대 대응 꼭짓점 거리 / 이미지 대각선이 `--tolerance`(기본 0.03) 이하이고 변환 결과가 생성되어야 성공입니다. 정답이 없는 이미지는 검출·변환 여부만 판정합니다.

## 검출 원리와 한계

전체 면적의 8% 이상인 윤곽선을 큰 순서로 검사하고, 둘레의 2% 오차로 근사한 볼록 사각형을 선택합니다. 네 점을 중심 각도순으로 정렬한 뒤 좌상단부터 배치하고, 마주 보는 변 길이로 출력 크기를 추정해 투시 변환합니다. 마지막으로 31×31 지역의 가우시안 가중 밝기에서 10을 뺀 임계값으로 이진화합니다.

그림자·반사·낮은 대비는 문서 경계를 끊고, 배경의 액자나 책상은 오검출을 유발할 수 있습니다. 높은 검출률이 올바른 문서 영역이나 글씨 판독성을 보장하지 않습니다. `04_contours.png`, `05_warped.png`, `06_result.png`를 함께 확인하세요. 실제 종이 종횡비, 구겨진 면, 글씨 방향은 복원하지 않습니다.

테스트는 좌표 순열·잘못된 사각형, 이진화, 파일 오류, headless CLI, GUI 제어 흐름과 카메라 자원 해제를 확인합니다. 실제 데스크톱 표시와 물리 카메라 동작은 별도 환경에서 확인해야 합니다.
