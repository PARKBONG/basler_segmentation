# finetuner

yolo26s-seg 파인튜닝 파이프라인. 코드/설정만 여기 두고, 실제 학습은 GPU 머신(B200 또는 RTX 5090)에서 돌립니다.

## 구성

| 파일 | 클래스 | config | 역할 |
|------|--------|--------|------|
| `download.py` | `Downloader` | `download_config.yaml` (자동) | 공개(Roboflow) 데이터셋 획득 |
| `preprocess.py` | `Preprocessor` | `preprocess_config.yaml` (자동) | 소스별 개별 처리(크롭·리사이즈·클래스 통일·분할·oversample) 후 하나로 병합 |
| `train.py` | `Trainer` | **`--config` 필수** | 파인튜닝 · 증강 미리보기 · ONNX export |
| `eval.py` | — | 명령행 인자 | 학습된 checkpoint 를 지정 split 에서 평가 |
| `common.py` | `Stage` | — | config 로드 · 경로 해석 · 로그 |
| `selftest.py` | — | — | 크롭 기하 · 폴리곤 클리핑 · 라벨 변환 · 누수 · config 규약 검증 |

```
python download.py  →  python preprocess.py  →  python train.py --config <yaml>
   datasets/rf_*/         datasets/processed/      runs/segment/ + 앱 Models/
```

각 단계는 import 해서 한 프로세스에서 이어 붙일 수도 있습니다:

```python
from download import Downloader; from preprocess import Preprocessor; from train import Trainer
Downloader().run(); Preprocessor().run(); Trainer("train_config.yaml").run()
```

### config 는 명시해야 합니다 (train)

`download`/`preprocess` 는 config 가 하나뿐이라 자기 이름의 yaml 을 자동으로 읽습니다.
**`train.py` 는 `--config` 없이는 실행되지 않습니다.**

```powershell
python train.py --config train_config.yaml            # 단일 스테이지
python train.py --config train_config.stage1.yaml     # 공개 warm-up
python train.py --config train_config.stage2.yaml     # in-domain 적응
```

후보가 여러 개인데 기본값을 고르면, 의도와 다른 설정으로 100 epoch 을 돌려도 알아챌 방법이
없습니다. 그래서 조용한 fallback 을 전부 없앴습니다 — 생략하면 `--config` 가 필수라는 오류와
함께 후보 목록이 나오고, 이름을 잘못 적어도 traceback 대신 후보를 알려줍니다. 그리고 어느
단계든 **읽은 config 의 절대경로를 로그 첫 줄에 찍습니다.**

같은 이유로 아래 항목들도 기본값 없이 오류를 냅니다:

| 빠진 것 | 결과 |
|---------|------|
| `stages` 블록 | 오류 (예전엔 train·export 가 켜진 것으로 간주됨) |
| `stages` 의 키 오타 (`expor: true`) | 오류 (조용히 무시하면 export 한 줄 알고 끝남) |
| `stages` 가 전부 `false` | 오류 (할 일이 없음) |
| `train.data` | 오류 (`datasets/processed` 로 추측하지 않음) |
| `train.model` | 오류 (COCO 로 되돌아가 stage1 warm-start 를 날리지 않음) |

`train.model` 에 경로 구분자가 있으면(`runs/segment/stage1/weights/best.pt`) 체크포인트
파일로 보고 `finetuner/` 기준으로 해석한 뒤 **존재를 확인**합니다. 구분자가 없으면
(`yolo26s-seg.pt`) Ultralytics 가 받아올 모델 이름이라 그대로 넘깁니다.

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

크롭 인자에 **공유 기본값은 없습니다** — 소스마다 자기 `crop` 블록이 전부입니다.
`raw` 는 풀사이즈라 640 크롭을 켜고, 공개셋은 이미 잘려 있어 `enabled: false` 로 두는 식입니다.

### 처리할 소스 고르기 (`targets`)

맨 위의 `targets:` 에 적은 이름만 처리합니다. 비우거나 지우면 전체입니다.

```yaml
targets:
  - raw
  - rf_a
```

산출물 폴더는 매 실행마다 비우고 다시 쓰므로, **`targets` 에 적은 소스만 학습셋에 남습니다**
(특정 소스만 빼고 다시 굽고 싶을 때 쓰세요). 없는 이름을 적으면 바로 오류로 알려줍니다.

## 크롭

**이미지와 폴리곤 라벨을 함께** 자릅니다. 위치 규약은 앱의 `FrameCropper.cs` 와 동일합니다 —
`center_x`(좌우)/`center_y`(상하) 가 0% 왼쪽/위 끝, 100% 오른쪽/아래 끝, 50% 중앙.

| 인자 | 뜻 |
|------|-----|
| `width` / `height` | 창 크기(px). **`0` 이면 그 축은 원본 전체** (원본보다 크면 원본으로 클램프) |
| `center_x` / `center_y` | 창 위치(%) — 좌우 / 상하 |
| `auto_crop` | 아래 참고. 켜면 위 네 값은 무시 |
| `min_area` | 크롭 후 남은 폴리곤 면적비 하한 |

- 크롭 창에 걸친 인스턴스는 잘리고, 남은 면적이 `min_area` 미만이면 그 인스턴스는 폐기됩니다.
- 살아남은 인스턴스가 하나도 없으면 그 이미지는 데이터셋에서 제외됩니다.
- `preview.enabled: true` 면 산출물 몇 장에 라벨을 그려 `processed/_preview/` 에 저장합니다 —
  크롭이 라벨을 제대로 따라 잘랐는지 눈으로 확인하세요.

### 자동 크롭 (`auto_crop`)

%로 위치를 고정하는 대신, **이미지마다 라벨(wire)에서 창을 직접 잡습니다.**

```yaml
      auto_crop:
        enabled: true
        margin: 0.25       # 라벨 경계상자 주위 여유 (경계상자 크기 대비)
        min_size: 640      # 창 한 변의 하한 = 경고 기준선
```

- 크기 = 라벨 경계상자 × `(1 + 2*margin)`, 가로·세로 **따로** 계산 → 가로로 긴 와이어는 가로만 넓어집니다.
- 그 값이 `min_size` 보다 작으면 `min_size` 로 키웁니다 (와이어가 가늘어도 창은 640 유지).
- 중심 = 라벨 경계상자 중심. 창이 이미지 밖으로 나가면 안쪽으로 밀어 넣습니다.
- **원본이 `min_size` 보다 작아** 창을 더 못 키우면 그 장수를 세어 마지막에 경고를 찍습니다.
- 라벨 폴리곤이 하나도 없는 이미지는 중심을 못 잡으므로 제외됩니다.

창 크기가 이미지마다 달라지므로, 크기를 맞추고 싶으면 `resize` 를 함께 쓰세요
(학습은 어차피 `imgsz` 로 letterbox 하므로 보통 `null` 로 둬도 됩니다).

## 증강

**offline 증강은 하지 않습니다.** Ultralytics 가 학습 중 매 epoch 새로 증강하므로
`train_config.yaml` 의 `train:` 블록 안 하이퍼파라미터(`hsv_*`, `scale`, `fliplr`, `mosaic`,
`copy_paste` …)로 조절합니다. 디스크에 구워두면 다양성이 오히려 줄고 온라인 증강과 이중으로 겹칩니다.

결과 확인 — 쓰는 config 의 `stages` 를 이렇게 두고 돌립니다:

```yaml
stages:
  preview_aug: true
  train: false
  export: false
```

`runs/preview/aug/train_batch*.jpg` 에 **학습이 실제로 먹는 배치**가 마스크까지 그려져 나옵니다
(증강 파이프라인을 따로 재현하지 않고 1 epoch·`fraction` 만큼만 태워 얻는 방식이라 어긋날 여지가 없음).

## 재학습 없이 export 만

쓰는 config 의 `stages` 를 이렇게 두고 `python train.py --config <그 yaml>`:

```yaml
stages: { preview_aug: false, train: false, export: true }
```

`train.project/name` 에서 `best.pt` 를 찾아 ONNX 로 변환하고, `export.deploy_to` 로 복사합니다
(기존 파일은 그대로 두고 타임스탬프를 붙인 새 파일로).

## 실행

학습 머신은 B200(sm_100) 또는 RTX 5090(sm_120) — 둘 다 Blackwell 세대라 **CUDA 12.8+ 빌드
PyTorch(cu128 휠)가 필수**이고, 같은 휠이 두 GPU를 모두 지원하므로 설치 명령은 동일합니다.
`train_config*.yaml` 의 `batch: -1`(AutoBatch)이 GPU 메모리에 맞춰 배치를 자동으로 잡아주므로
머신이 바뀌어도 config 는 수정할 필요 없습니다.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python selftest.py       # 파이프라인 자체 검증 (GPU 불필요)
python preprocess.py
python train.py --config train_config.yaml
```

torch 가 GPU 를 제대로 잡았는지 확인:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## roboflow
$ setx ROBOFLOW_API_KEY "mzbA71wxvqdAFlyB6nPN" ## dummy fake key
