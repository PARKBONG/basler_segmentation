# basler_segmentation

Basler 카메라(eye-in-hand)로 WAAM 와이어를 실시간 분할하는 프로젝트. 세 덩어리로 나뉜다.

```
basler_segmentation/
├─ datasets/   데이터 (git 밖)          앱이 녹화하고, training 이 라벨·전처리·학습에 쓴다
├─ training/   학습 Python (uv 프로젝트) 라벨러 · 전처리 · 학습 · 평가 · 추론
└─ app/        실증 C# (.NET 8 WPF)      라이브 뷰 · 녹화 · 학습한 모델로 실시간 추론
```

흐름: `app` 으로 녹화 → `training/labeller` 로 라벨 → `training` 으로 전처리·학습 → 쓸 모델을 `weights/` 에
보관하고 `weights/deploy.yaml` 에서 고름 → `app` 빌드가 그 ONNX 를 넣어 실증.

## datasets/

데이터는 용량 때문에 git 에 넣지 않는다 (`.gitignore`). 예외로 샘플 1장
(`raw/kimm_260928_204214/` 의 `000001__…` 이미지 + `labels/` · `labels_seg/`)와 기준 레이어 4개
(`reference/reference_{cutout,outline,tube,wire}.png` — 앱 기준 오버레이 · 라벨러 기준 와이어)만 들어 있어,
새로 clone 한 곳에서도 앱·라벨러·전처리가 바로 돈다 — 실제 데이터는 원본 PC 의 `datasets/` 를 통째로
복사해 온다. 폴더 규약:

| 경로 | 내용 | 만드는 쪽 |
|---|---|---|
| `raw/kimm_YYMMDD_HHMMSS[_메모]/images/` | 녹화 세션 원본 (크롭 전 풀사이즈 PNG) | 앱 REC |
| `raw/kimm_*/labels/` · `labels_seg/` | 손으로 잡은 OBB · 와이어 seg 폴리곤 | 라벨러 |
| `raw/_removed_black/` · `_removed_manual/` | 빼 둔 프레임 (되돌리기용) | 정리 작업 · 라벨러 삭제 |
| `processed_seg/` · `processed_obb/` | 학습용 산출물 (크롭·분할) | `training/preprocess.py` |
| `reference/` | 기구부(튜브+와이어) 기준 이미지·누끼·윤곽선 | `training/tools/make_reference_cutout.py` |

## training/

```powershell
cd training
uv sync --extra train --extra export      # 첫 설치 (torch cu128 · ultralytics · ONNX · pytest · ruff)
uv run labeller/server.py                 # 라벨러 → http://localhost:8765
uv run preprocess.py                      # raw → processed_seg
uv run train.py --config train.single.yaml
uv run pytest                             # 파이프라인 테스트 (GPU·데이터 불필요)
```

자세한 건 [training/README.md](training/README.md). VS Code 에서는 `Ctrl+Shift+P → Tasks: Run Task` 의
`training: train` / `training: inference — …` / `training: tensorboard` 로 돌릴 수 있다.

## app/

```powershell
# app/external/YoloDotNet — 빌드에 쓰는 두 폴더만 받는다 (~1MB).
# 그냥 `git submodule update --init` 하면 테스트용 ONNX 모델까지 ~500MB 를 받는다.
git submodule init
git clone --filter=blob:none --no-checkout https://github.com/NickSwardh/YoloDotNet.git app/external/YoloDotNet
git -C app/external/YoloDotNet sparse-checkout set YoloDotNet YoloDotNet.ExecutionProvider.DirectML
git submodule update --force
git submodule absorbgitdirs

dotnet build app/basler_segmentation.sln
```

`app/external/` 에 Basler pylon SDK(필요한 부분만 벤더링)와 YoloDotNet(submodule)이 있다.
설정은 [app/src/BaslerLiveView/config/config.xml](app/src/BaslerLiveView/config/config.xml) —
녹화 폴더·기준 이미지 같은 상대경로는 저장소 루트(`.git` 이 있는 폴더) 기준.
추론 모델은 `<Inference><Model>` — seg / obb 모델을 파일에서 알아서 구분한다. 그 파일은 빌드가
[weights/deploy.yaml](weights/deploy.yaml) 에서 고른 보관본을 복사한 것 (clone 뒤 `git lfs pull` 필요).
자세한 건 [app/src/BaslerLiveView/README.md](app/src/BaslerLiveView/README.md).

## 저장소 규칙

- 줄바꿈 `.gitattributes`, 들여쓰기·인코딩 `.editorconfig`, Python 린트 `training/pyproject.toml` (ruff).
- CI(`.github/workflows/ci.yml`): push 마다 `training` ruff · pytest, `app` 빌드 · `dotnet format` 확인.
