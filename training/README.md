# training

와이어 seg / obb 모델(yolo26s) 학습 파이프라인 — 라벨링부터 앱 배포까지. uv 프로젝트 하나.

```
labeller/  →  preprocess.py  →  train.py (single)  →  eval.py  →  train.py (seg_all · obb_all)  →  app Models/
 라벨링        processed_seg/    측정 · best epoch     test 1회     전체 데이터로 재학습              ONNX 배포
               processed_obb/
```

## 구성

| 경로 | 역할 |
|---|---|
| `labeller/` | 브라우저 라벨러 (`server.py` · `index.html` · 끝 경계 `edge.py`) |
| `preprocess.py` | 원본 → 학습 데이터셋 (크롭 · 클래스 통일 · 세션 단위 분할 · seg→obb 변환) |
| `train.py` | 학습 · 증강 미리보기 · ONNX export · 앱 배포 |
| `eval.py` | checkpoint 를 지정 split 에서 평가 + 정답/추론 겹친 그림 |
| `infer.py` | 사진 한 장 → 예측 그림, 세션 폴더 → 동영상 |
| `epoch_overlay.py` | 학습 중 매 val 마다 원본 4장 + 증강 12장 추론 격자 |
| `common.py` | config 로드(+ `base:` 상속) · 경로 해석 · 로그 |
| `tools/make_reference_cutout.py` | 기구부 기준 이미지 → 누끼·윤곽선 (앱·라벨러 오버레이) |
| `configs/` | `preprocess.yaml`, `train.*.yaml` (+ 공통 `base/train.yaml`), `eval.*.yaml` |
| `tests/` | pytest — 합성 데이터로 파이프라인 전체 검증 (GPU·데이터 불필요) |

## 설치 · 실행

환경은 uv 하나 — `pyproject.toml` + `uv.lock`, 파이썬은 `.python-version` 의 3.12.
학습 머신은 Blackwell(B200 · RTX 5090)이라 **CUDA 12.8 빌드 torch** 를 쓴다 (pyproject 가 인덱스 고정).

```powershell
cd training
uv sync --extra train --extra export        # torch(cu128) · ultralytics · ONNX + dev(pytest · ruff)
uv run labeller/server.py                   # 라벨러 → http://localhost:8765
uv run preprocess.py                        # → ../datasets/processed_seg/
uv run preprocess.py --task obb --out ../datasets/processed_obb   # OBB 학습할 때만
uv run train.py --config train.single.yaml
uv run eval.py --config eval.single.yaml
uv run infer.py --weights runs/segment/yolo26s-seg-finetune/weights/best.pt --source ../datasets/raw/kimm_261019_161818
```

> **`uv sync` 는 항상 `--extra train --extra export` 와 함께.** extra 없이 돌리면 uv 가 환경을
> 정확히 맞추느라 torch 등을 지운다 (라벨링만 하는 머신이면 그게 의도).

VS Code: `Ctrl+Shift+P → Tasks: Run Task` 의 `training: train` · `training: inference — 사진 1장 /
세션 → 동영상` · `training: tensorboard` (`.vscode/tasks.json`).

## 라벨링 (`labeller/`)

카메라가 기구에 붙어(eye-in-hand) 튜브·와이어가 화면에서 늘 같은 자리라, 와이어 박스를 그리지 않고
기준 자세에서 **우측 상단 꼭짓점(튜브 출구)을 축으로 옮기고 돌려** 맞춘다.

- 세션별 보기 (↑↓ 세션, ←→ 이미지), 앱과 같은 기구부 오버레이, 크롭 창 조절 → `preprocess.yaml` 에 저장
- 일괄 덮어쓰기: 세션 전체 / 이 번호부터 끝까지 / 이 번호부터 N번까지
- 끝 경계: **Compute edge 1** (OBB 윗변·옆변 + 영상에서 찾은 끝 곡선) 또는 **OBB as edge** →
  `labels_seg/` 에 와이어 폴리곤 저장. OBB 를 바꾼 장은 seg 가 지워지고 edge 1 로 다시 계산
- 이미지 삭제(Shift+Delete) → `datasets/raw/_removed_manual/` 로 이동 (되돌릴 수 있음)

산출물 (세션 폴더마다): `labels/` = 손으로 잡은 OBB, `labels_seg/` = 학습에 쓰는 와이어 폴리곤.

## 데이터 · 전처리 (`preprocess.py`, `configs/preprocess.yaml`)

```
datasets/
  raw/kimm_YYMMDD_HHMMSS[_메모]/   앱 REC 세션 — images/ + labels/ + labels_seg/
  processed_seg/                 preprocess.py 산출물 (task: seg)
    data.yaml                    전 소스 통합본 (train.single · eval.single)
    kimm/images|labels/{train,val,test}, preview/, data.yaml (eval.final), data.all.yaml (seg_all)
  processed_obb/                 같은 소스를 --task obb 로 구운 것 (obb_all)
  reference/                     기구부 기준 이미지·누끼·윤곽선
```

`sources:` 항목 하나가 데이터셋 하나:

| 인자 | 뜻 |
|---|---|
| `path` | 소스 폴더. 마지막 칸 글롭(`../datasets/raw/kimm_*`)이면 세션 폴더 전부를 한 소스로 — 세션이 늘어도 config 그대로. 빼고 싶은 세션은 폴더 이름을 `kimm_` 로 시작하지 않게 |
| `labels` | 라벨 폴더 이름 (기본 `labels`). kimm 은 `labels_seg` |
| `names` | 폴더에 `data.yaml` 이 없을 때 쓰는 원본 클래스 이름 |
| `class_map` | 원본 클래스명 → 최종 클래스명. 여기 없는 클래스는 버림 |
| `type` | 원본 라벨 형식 `seg` / `obb` — `task: obb` 로 구울 때 seg 폴리곤에 OBB 를 다시 맞춤 |
| `group_by` | `session` = 세션 폴더 통째로 한 split 에 (2fps 연속 프레임끼리 근중복이라 파일 단위로 나누면 val 이 train 의 복사본이 됨) |
| `val_sessions` / `test_sessions` | 세션 직접 지정 — 지금은 장면 묶음(10/14 · 10/19 격자 · 10/20 hollow)이 train 에 모두 들어가게 골라 둠 |
| `val_ratio` / `test_ratio` | 세션 지정이 없을 때 쓰는 비율 (세션 단위라 근사) |
| `oversample` · `crop` · `resize` | train 복제 배수 · 크롭(아래) · 크롭 후 리사이즈 |

- `targets: [이름]` 에 적은 소스만 다시 굽는다 (다른 소스 산출물 유지).
- 원본에서 **이미지만 지우면** 다음 실행 때 짝 라벨도 지운다 (이미지 0장 폴더는 경로 실수로 보고 보호).
- 크롭은 이미지와 폴리곤을 **함께** 자른다. 위치 규약(0% = 왼쪽/위 끝, 100% = 오른쪽/아래 끝)과 크기
  `0` = "그 축 원본 전체" 는 앱 `FrameCropper.cs` 와 같다. `auto_crop` 을 켜면 이미지마다 라벨에서 창을
  잡는다. 창 밖으로 나간 인스턴스는 잘리고, 남은 면적이 `min_area` 미만이면 버린다.
- `preview` 를 켜면 소스별 `preview/` 에 라벨을 그린 산출물 몇 장 — 크롭이 라벨을 따라갔는지 확인용.

## 학습 (`train.py`, `configs/train.*.yaml`)

| config | 모델 | 데이터 | 종료 | 배포 |
|---|---|---|---|---|
| `train.single.yaml` | seg | train 세션 / val·test 분리 | early stop (patience 50) | 안 함 — 측정용 |
| `train.seg_all.yaml` | seg | 전체 (`data.all.yaml`) | epochs = single 의 best epoch | `Models/yolo26s-seg.onnx` |
| `train.obb_all.yaml` | obb | 전체 (`processed_obb`) | epochs = single 의 best epoch | `Models/yolo26s-obb.onnx` |

- **순서**: `single` 로 성능을 재고(진짜 val 로 early stop), 그 best epoch 를 `*_all` 의 `epochs` 에 넣어
  데이터를 다 넣고 다시 학습한다. `*_all` 은 val 이 train 과 겹쳐 early stop 이 과적합을 못 잡기 때문.
- **공통 설정**은 `configs/base/train.yaml` — 각 config 는 `base: base/train.yaml` 로 상속하고 다른 값만
  적는다 (dict 는 키별로 덮어씀). 증강을 바꾸면 세 학습에 같이 반영된다.
- **증강**: 회전 10°(실제 각도 p99 7°) · 이동 0.1 · 확대 0.2 · 밝기 0.6 · 블러 10%(albumentations) · mosaic 50%. 좌우/상하
  반전은 끔 (실제로 없는 장면). offline 증강은 하지 않는다 — Ultralytics 가 매 epoch 새로 증강.
- **보기**: TensorBoard(`uv run tensorboard --logdir runs` 또는 task) 와, 매 val 마다
  `<실행 폴더>/overlays/epoch_NNN.jpg` (원본 4장 + 증강 12장, 증강 값 표시, 초록 = 정답). 끝나면 `best.jpg`.
- **증강 미리보기**: `stages: {preview_aug: true, train: false, export: false}` → `runs/preview/aug/`.
- **export 만**: `stages: {preview_aug: false, train: false, export: true}` — 최신 실행의 best.pt 를 ONNX 로.
- **배포**(`export.deploy_to`): 앱 `Models/<이름>.onnx` 를 덮어쓰고(앱 `config.xml` 의 `<Model>` 이 이 이름)
  `Models/history/<이름>.<시각>.onnx` 로 이력을 남긴다. 앱은 모델 파일을 보고 seg/obb 를 알아서 고른다.
- 결과는 `runs/<task>/<name>/<YYMMDD_HHMMSS>/` — 실행마다 쌓인다. config 에서 타임스탬프를 뺀
  `runs/<task>/<name>/weights/best.pt` 는 그 name 의 최신 실행으로 해석된다.

## 평가 · 추론

- `eval.single.yaml` — val 세션 (학습 중 best 선택에 쓰인 것).
- `eval.final.yaml` — **test 세션, 측정용 모델에 딱 한 번.** 이 결과를 보고 설정을 다시 만지면 test 가
  오염된다.
- 그림: `eval/<시각>__<가중치>__<데이터셋>-<split>/` — 초록 = 정답, 빨강 = 추론(seg 는 마스크 윤곽),
  점선 = 가장 아래 두 점의 y (와이어 끝 높이 오차).
- `infer.py` — 크롭(`auto` = preprocess.yaml 값, `W H CX CY`, `none`) 후 예측. 사진이면 `<이름>_pred.jpg`,
  세션 폴더면 `<세션>.mp4` (`runs/infer/<시각>/`).

## config 규약 — 조용한 기본값 없음

- 후보가 여럿인 스크립트(`train.py`, `eval.py`)는 `--config` 없이 돌지 않는다. 후보가 하나뿐이면
  (`preprocess.yaml`) 생략 가능. 읽은 config 경로는 항상 로그 첫 줄에 찍힌다.
- `stages` 블록·키 오타·전부 false, `train.data`·`train.model` 누락은 추측하지 않고 오류.
- `train.model` 에 경로 구분자가 있으면 체크포인트로 보고 존재를 확인, 없으면(`yolo26s-seg.pt`)
  Ultralytics 가 받아올 이름.
- yaml 안의 경로는 `training/` 기준.

## 테스트 · 린트

```powershell
uv run pytest            # tests/ — 합성 데이터, GPU·실제 데이터 불필요
uv run ruff check .      # 설정은 pyproject.toml (포맷터는 안 씀 — 줄 끝 주석 정렬 유지)
```

CI(`.github/workflows/ci.yml`)가 push 마다 둘 다 돌리고, 앱 빌드·포맷도 확인한다.

## 같은 규칙이 여러 곳에 있는 것

언어가 달라 공유할 수 없어 같은 규칙을 따로 구현한 곳들 — **하나를 바꾸면 나머지도 맞춘다.**

| 규칙 | 위치 |
|---|---|
| 크롭 창 계산 (크기 0 = 전체, 위치 0~100%) | `preprocess.crop_rect` · 앱 `FrameCropper.cs` · 라벨러 `index.html` `cropRect()` |
| 와이어 자세 (긴 변 = 축, 우측 상단 꼭짓점 = 피벗) | `labeller/edge.obb_pose` · 라벨러 `index.html` `poseFromPoints()` |
| 증강 범위 | `configs/base/train.yaml` → 학습(Ultralytics)과 `epoch_overlay.py`(값 표시용 재현) |
