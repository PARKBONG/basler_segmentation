# train — YOLO26s-seg 2단계 파인튜닝 파이프라인

WAAM 영상에서 **wire** instance segmentation 을 학습합니다.
코드/설정만 여기(로컬)에서 작성하고, 실제 학습은 RTX 5090 머신에서 돌립니다.

## 전략: 2단계 순차 파인튜닝 (staged fine-tuning)

```
COCO 사전학습 (yolo26s-seg.pt)
      │
      ▼  [Stage 1] 공개 데이터로 warm-up → "와이어" 일반 특징 습득
      │            (in-domain 미포함 = 데이터 위생)
      ▼  [Stage 2] in-domain(내 영상)으로 도메인 적응 (낮은 LR, backbone freeze)
      │            checkpoint 는 stage2 config 에 직접 지정 (수동 체이닝)
      ▼
   최종 best.pt → ONNX → 앱(BaslerLiveView) Models/
```

## 파일

| 파일 | 역할 |
|------|------|
| `download.py` / `merge_config.yaml` | (선택) Roboflow 공개셋 SDK 다운로드 |
| `merge.py` / `merge_config.yaml` | stage1(공개)·stage2(in-domain) 데이터셋 2벌 생성 |
| `train.py` / `train_config.stage1.yaml` / `train_config.stage2.yaml` | 단일 stage 학습 (`--config` 로 선택) |
| `eval.py` | checkpoint 를 지정 split(val/test)에 평가 (Box/Mask mAP) |
| `export.py` / `export_config.yaml` | best.pt → ONNX 변환 후 앱에 비파괴 배치 |

## 실행 순서

```bash
pip install -r requirements.txt          # 5090: torch 는 cu128 인덱스로 먼저 (requirements.txt 참고)

# 1) 공개셋 다운로드 (선택)
python download.py

# 2) 데이터 생성 → datasets/stage1(공개), datasets/stage2(in-domain, 실험 단위 3분할)
python merge.py

# 3) Stage 1 (공개 warm-up)
python train.py --config train_config.stage1.yaml

# 4) train_config.stage2.yaml 의 model: 에 stage1 checkpoint 경로 기입
#    예) runs/segment/stage1/weights/best.pt

# 5) Stage 2 (in-domain 적응)
python train.py --config train_config.stage2.yaml

# 6) ONNX export → 앱 Models/ (타임스탬프 붙여 비파괴)
python export.py
```

## 데이터 위생 (반드시 지킬 것)

- **val/test 는 100% in-domain.** 공개셋은 stage1 train 전용, val/test 에 절대 안 들어감.
- **실험(영상) 단위 분할.** in-domain 은 영상 프레임이라, `merge.py` 가 `images/<실험>/`
  폴더를 통째로 train/val/test 에 배정 → 상관-프레임 누수(leakage) 차단.
- **test 는 최종 1회만.** 학습·checkpoint 선택에 절대 사용 금지. test 수치를 보고
  하이퍼파라미터를 다시 만지면 그 순간 오염 → 논문에 못 씀.
  개발 중 비교는 항상 **val** 로.

## Ablation — 2단계의 가치 측정

Stage1·Stage2 checkpoint 를 **같은 in-domain 데이터**에 평가해 도메인 적응 효과를 정량화합니다.
Stage1 은 in-domain 을 학습에 쓰지 않았으므로, in-domain val/test 는 Stage1 입장에서도
완전한 held-out (누수 없음).

```bash
# 개발 중: in-domain val 로 진행 상황/비교 (여러 번 봐도 됨)
python eval.py --weights runs/segment/stage1/weights/best.pt \
               --data ../datasets/stage2/data.yaml --split val
python eval.py --weights runs/segment/stage2/weights/best.pt \
               --data ../datasets/stage2/data.yaml --split val

# 최종(논문용): in-domain test — 확정된 두 모델에 딱 한 번씩만
python eval.py --weights runs/segment/stage1/weights/best.pt \
               --data ../datasets/stage2/data.yaml --split test
python eval.py --weights runs/segment/stage2/weights/best.pt \
               --data ../datasets/stage2/data.yaml --split test
```

기대 결과 (논문 표):

| 모델 | in-domain test | 의미 |
|------|:---:|------|
| Stage1 (공개만) | 낮음 | 도메인 적응 **전** (zero-shot to WAAM) |
| Stage2 (적응 후) | 높음 | 도메인 적응 **후** |

이 델타(Δ)가 2단계 파인튜닝이 필요했다는 근거입니다.

> 추가 ablation 아이디어: Stage2 의 `freeze: 10`(backbone 고정) vs `freeze: 0`(full)
> 를 각각 학습해 in-domain test 로 비교. (test 는 최종 확정 모델들에 한 번씩만.)
