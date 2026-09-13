# Document Digitization Pipeline

## 프로젝트 소개

문서 사진을 반듯한 흑백 스캔 이미지로 바꾸는 작은 Python 프로그램. OpenCV와 NumPy만 사용합니다. OCR은 하지 않습니다.

## 핵심 특징

- 문서 경계 검출, 원근 보정, 적응형 이진화
- 슬라이더로 파라미터 조절, 처리 단계별 이미지 저장
- GUI 없이 실행하거나 생성 이미지 20장으로 설정 비교

## 아키텍처

```text
이미지 → 회색조·블러 → Canny → 사각형 검출 → 원근 보정 → 이진화
```

핵심 로직은 `scanner/pipeline.py`, 입출력은 `scanner/io.py`, 명령행 실행은 `scanner/cli.py`, GUI는 `scanner/gui.py`에 있습니다. 평가 도구도 같은 `scan()` 함수를 사용합니다.

## 실행

Python 3.8 이상.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --image data/simple_1.png
```

Windows에서는 가상환경을 `.venv\Scripts\activate`로 활성화합니다.

한 창에서 여섯 처리 단계와 조절 패널을 확인합니다. 슬라이더를 클릭하거나 드래그하면 블러 커널 크기와 Canny 임계값이 즉시 반영됩니다. 표시되는 값은 실제 적용 값이며, Canny 하한은 항상 상한보다 작게 유지됩니다.

`Tab`으로 조절 항목을 선택하고 `+` / `-`로 미세 조정합니다. `s`로 저장, `q` 또는 Esc로 종료하며 창의 닫기 버튼으로도 종료할 수 있습니다. 검출 여부와 저장 상태는 창 하단에 표시됩니다.

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

`tools/manifest.json`에 등록된 GPT 생성 이미지 20장을 세 가지 설정으로 비교합니다. 사진별 밝기·배치·배경 메모는 이 파일에서 관리합니다. 문서의 네 꼭짓점을 제대로 잡아 반듯하게 보정했는지 직접 확인하세요. 사각형 검출 여부인 `detected`는 성공 판정이 아닙니다.

집계는 `report.md`를 덮어씁니다. 분석 메모는 별도 파일에 보관하고, 재실험에는 새 출력 폴더를 사용하세요. 추가 옵션은 `python -m tools.evaluate --help`로 확인합니다.

## 한계

그림자·반사·복잡한 배경에서 경계를 놓치거나 다른 사각형을 잡을 수 있습니다. 실제 종이의 종횡비, 구겨진 면, 글씨 방향은 복원하지 않습니다.
