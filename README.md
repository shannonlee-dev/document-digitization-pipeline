# Document Digitization Pipeline

## 프로젝트 소개

문서 사진을 반듯한 흑백 스캔 이미지로 바꾸는 작은 Python 프로그램. OpenCV와 NumPy만 사용합니다. OCR은 하지 않습니다.

## 핵심 특징

- 문서 경계 검출, 원근 보정, 적응형 이진화
- 트랙바로 파라미터 조절, 처리 단계별 이미지 저장
- GUI 없이 실행하거나 샘플 20장으로 설정 비교

## 아키텍처

```text
이미지 → 회색조·블러 → Canny → 사각형 검출 → 원근 보정 → 이진화
```

핵심 로직은 `scanner/pipeline.py`, 입출력은 `scanner/io.py`, 실행과 GUI는 `scanner/cli.py`에 있습니다. 평가 도구도 같은 `scan()` 함수를 사용합니다.

## 실행

Python 3.8 이상.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --image data/simple_1.png
```

Windows에서는 가상환경을 `.venv\Scripts\activate`로 활성화합니다.

트랙바로 블러와 Canny 임계값을 조절합니다. `s`로 저장, `q` 또는 Esc로 종료합니다.

```bash
# 화면 없이 처리하고 저장
python main.py --image data/simple_1.png --headless --output outputs/scan

# 전체 옵션
python main.py --help
```

결과는 `outputs/scan/`에 단계별로 저장됩니다. 같은 경로는 덮어쓰며, 검출 실패 시 진단 이미지만 남기고 이전 보정·이진화 결과는 삭제합니다.

## 평가와 테스트

```bash
python -m tools.evaluate --output outputs/evaluation
# results.csv의 success에 true/false를 입력한 뒤 집계
python -m tools.evaluate --output outputs/evaluation --summarize
python -m unittest discover -s tests -v
```

`data/`의 20장을 세 가지 설정으로 비교합니다. 문서의 네 꼭짓점을 제대로 잡아 반듯하게 보정했는지 직접 확인하세요. 사각형 검출 여부인 `detected`는 성공 판정이 아닙니다.

집계는 `report.md`를 덮어씁니다. 분석 메모는 별도 파일에 보관하고, 재실험에는 새 출력 폴더를 사용하세요. 추가 옵션은 `python -m tools.evaluate --help`로 확인합니다.

## 한계

그림자·반사·복잡한 배경에서 경계를 놓치거나 다른 사각형을 잡을 수 있습니다. 실제 종이의 종횡비, 구겨진 면, 글씨 방향은 복원하지 않습니다.
