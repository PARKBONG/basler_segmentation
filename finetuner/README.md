# finetuner

yolo26s-seg 파인튜닝 파이프라인. 코드/설정만 여기 두고, 실제 학습은 RTX 5090 머신에서 돌립니다.

## 구성

각 단계는 **자기 이름의 config 하나**만 읽습니다 (명령행 인자 없음).

| 파일 | 클래스 | config | 역할 |
|------|--------|--------|------|
| `download.py` | `Downloader` | `download_config.yaml` | 공개(Roboflow) 데이터셋 획득 |
| `preprocess.py` | `Preprocessor` | `preprocess_config.yaml` | 소스별 개별 처리(크롭·리사이즈·클래스 통일·분할·oversample) 후 하나로 병합 |
| `train.py` | `Trainer` | `train_config.yaml` | 파인튜닝 · 증강 미리보기 · ONNX export |
| `common.py` | `Stage` | — | config 로드 · 경로 해석 · 로그 |
| `selftest.py` | — | — | 크롭 기하 · 폴리곤 클리핑 · 라벨 변환 · 누수 검증 |

```
python download.py     →  python preprocess.py  →  python train.py
   datasets/rf_*/            datasets/processed/       runs/segment/ + 앱 Models/
```

각 단계는 import 해서 한 프로세스에서 이어 붙일 수도 있습니다:

```python
from download import Downloader; from preprocess import Preprocessor; from train import Trainer
Downloader().run(); Preprocessor().run(); Trainer().run()
```

## datasets/ 레이아웃

전부 `.gitignore` 대상입니다 (용량).

```
datasets/
  raw/         내 카메라 원본 (인도메인)
    images/train/    앱(BaslerLiveView)의 REC 가 쌓는 크롭 전 풀사이즈 PNG
    labels/train/    라벨링 결과 (YOLO seg 폴리곤)
    data.yaml
  rf_*/        download.py 가 받은 공개셋
  processed/   preprocess.py 산출물 = train.py 가 읽는 곳 (+ _preview/)
```

`raw` 의 라벨링만 수동 단계입니다. `raw/data.yaml` 은 두 줄이면 됩니다:

```yaml
names: {0: wire}
train: images/train
```

앱은 **크롭하지 않은 원본**을 저장합니다. 크롭은 폴리곤 라벨까지 함께 잘라야 하므로
`preprocess.py` 가 담당하고, 앱에서 버린 픽셀은 되돌릴 수 없기 때문입니다. 앱 툴바
슬라이더로 눈으로 찾은 위치(%)를 `preprocess_config.yaml` 의 `crop.center_x/center_y` 에
그대로 옮겨 적으면 됩니다 — 두 곳이 같은 0~100% 규약을 씁니다.

## 소스별 개별 처리

`preprocess_config.yaml` 의 `sources:` 항목 하나가 곧 데이터셋 하나이고, 인자를 각자 가집니다:

| 인자 | 뜻 |
|------|-----|
| `path` | 소스 폴더 (`data.yaml` + `images/…` + `labels/…`) |
| `class_map` | 소스클래스명 → 최종클래스명. 여기 없는 클래스는 버림 |
| `val_ratio` | 이 소스에서 val 로 뗄 비율. 공개셋은 `0.0` (검증 오염 방지) |
| `oversample` | train 쪽 물리 복제 배수 (val 에는 적용 안 됨) |
| `crop` | 아래 참고. 소스마다 켜고 끌 수 있음 |
| `resize` | 크롭 후 리사이즈. `null` 이면 그대로 |

`defaults.crop` 은 모든 소스의 기본값이고, 소스의 `crop` 이 그 위에 덮어씁니다.
예를 들어 `raw` 는 풀사이즈라 640 크롭을 켜고, 공개셋은 이미 잘려 있어 끄는 식입니다.

## 크롭

**이미지와 폴리곤 라벨을 함께** 자릅니다. 위치 규약은 앱의 `FrameCropper.cs` 와 동일합니다 —
`center_x`/`center_y` 가 0% 왼쪽/위 끝, 100% 오른쪽/아래 끝, 50% 중앙.

- 크롭 창에 걸친 인스턴스는 잘리고, 남은 면적이 `min_area` 미만이면 그 인스턴스는 폐기됩니다.
- 살아남은 인스턴스가 하나도 없으면 그 이미지는 데이터셋에서 제외됩니다.
- `preview.enabled: true` 면 산출물 몇 장에 라벨을 그려 `processed/_preview/` 에 저장합니다 —
  크롭이 라벨을 제대로 따라 잘랐는지 눈으로 확인하세요.

## 증강

**offline 증강은 하지 않습니다.** Ultralytics 가 학습 중 매 epoch 새로 증강하므로
`train_config.yaml` 의 `train:` 블록 안 하이퍼파라미터(`hsv_*`, `scale`, `fliplr`, `mosaic`,
`copy_paste` …)로 조절합니다. 디스크에 구워두면 다양성이 오히려 줄고 온라인 증강과 이중으로 겹칩니다.

결과 확인:

```yaml
stages:
  preview_aug: true
  train: false
  export: false
```

`runs/preview/aug/train_batch*.jpg` 에 **학습이 실제로 먹는 배치**가 마스크까지 그려져 나옵니다
(증강 파이프라인을 따로 재현하지 않고 1 epoch·`fraction` 만큼만 태워 얻는 방식이라 어긋날 여지가 없음).

## 재학습 없이 export 만

```yaml
stages: { preview_aug: false, train: false, export: true }
```

`train.project/name` 에서 `best.pt` 를 찾아 ONNX 로 변환하고, `export.deploy_to` 로 복사합니다
(기존 파일은 그대로 두고 타임스탬프를 붙인 새 파일로).

## 실행

```powershell
pip install -r requirements.txt
python selftest.py       # 파이프라인 자체 검증 (GPU 불필요)
python preprocess.py
python train.py
```
