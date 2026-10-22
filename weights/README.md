# weights/

남겨 둘 학습 결과만 골라 두는 곳. `training/runs/` 는 git 밖이라 지우거나 PC 를 바꾸면
사라지므로, 관심 있는 실행만 여기로 복사해 git 으로 관리한다. 여기의 `.pt` · `.onnx` 만
**Git LFS** 로 올라간다 (`.gitattributes`) — clone 한 뒤 `git lfs pull` 이 필요할 수 있다.

모양은 `training/runs/<task>/` 아래와 같다 — `<name>/<YYMMDD_HHMMSS>/`:

```
weights/yolo26s-seg-all/261022_151611/
├─ weights/best.pt      학습 재개 · eval.py · infer.py --weights 로 그대로 · 다시 export
├─ weights/last.pt      마지막 epoch (끝난 실행은 옵티마이저가 빠져 best 와 같은 크기)
├─ weights/best.onnx    앱용 (Models/ 로 복사). 메타데이터에 train_commit · train_dirty · train_run
├─ args.yaml            Ultralytics 학습 인자 전체
├─ results.csv          epoch 별 손실 · 점수
├─ run_info.yaml        학습 시작 시점의 커밋 · 커밋 안 된 수정 · config · 데이터 장수
└─ code.patch           학습 당시 커밋 안 된 수정이 있었으면 (git apply 로 재현)
```

손으로 옮겨도 되고, `training/` 에서 `uv run archive.py runs/<task>/<name>/<시각>` 을 쓰면
위 파일만 골라 복사하고 run_info.yaml 과 .onnx 출처 메타데이터를 채워 준다.
`epoch*.pt` 는 넣지 않는다 (옵티마이저까지 들어 있어 ~115 MB). VS Code task
`archive — weights/ 에 보관` 으로도 된다 (runs/ 의 best.pt 를 고르면 그 실행을 보관).

`*_all` (전체 데이터) 실행의 점수는 이미 학습한 데이터로 채점한 것이라 성능 지표가 아니다 —
성능은 같은 설정의 `train.single` 실행으로 본다.
