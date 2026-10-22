"""
학습 실행 하나를 저장소 루트 weights/ 에 보관 (git · Git LFS 로 관리).

    uv run archive.py runs/segment/yolo26s-seg-all/261022_151611
    uv run archive.py runs/obb/yolo26s-obb-all/261022_153600

→ ../weights/<name>/<시각>/  — runs/<task>/ 아래와 같은 모양 (예: weights/yolo26s-seg-all/261022_151611/)
    weights/best.pt    학습 재개 · eval · infer(--weights 로 그대로) · 다시 export 용
    weights/last.pt    마지막 epoch (끝난 실행은 옵티마이저가 빠져 best 와 같은 크기)
    weights/best.onnx  앱용 (출처 메타데이터 포함 — provenance.stamp_onnx)
    args.yaml      Ultralytics 학습 인자 전체
    results.csv    epoch 별 손실 · 점수
    run_info.yaml  커밋 · 커밋 안 된 수정 · config · 데이터 장수 (없던 예전 실행은 .pt 의 git 정보)
    code.patch     학습 당시 커밋 안 된 수정이 있었으면

runs/ 은 그대로 둔다 (복사) — infer · eval 의 '최신 실행' 찾기와 task 드롭다운이 runs/ 를 본다.
epoch*.pt 는 옮기지 않는다 (옵티마이저까지 들어 있어 ~115 MB).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

import provenance
from common import HERE, resolve

DEST = HERE.parent / "weights"
FILES = ("weights/best.pt", "weights/last.pt", "weights/best.onnx", "args.yaml", "results.csv", provenance.PATCH)


def main() -> None:
    ap = argparse.ArgumentParser(description="학습 실행을 저장소 루트 weights/ 에 보관")
    ap.add_argument("run", help="실행 폴더 (예: runs/segment/yolo26s-seg-all/261022_151611) "
                                "— 그 안의 weights/ 나 best.pt 를 줘도 됨 (task 드롭다운)")
    ap.add_argument("--force", action="store_true", help="이미 보관한 폴더를 덮어씀")
    args = ap.parse_args()

    run = (Path(args.run) if Path(args.run).exists() else resolve(args.run)).resolve()
    if run.is_file():                  # .../<시각>/weights/best.pt
        run = run.parent
    if run.name == "weights":          # .../<시각>/weights
        run = run.parent
    best = run / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"best.pt 가 없습니다 (학습이 아직 안 끝났거나 경로가 틀림): {best}")
    if not (run / "weights" / "best.onnx").exists():
        raise SystemExit(f"best.onnx 가 없습니다 — export 먼저 (stages: train false, export true): {run}")

    dest = DEST / run.parent.name / run.name
    if dest.exists() and not args.force:
        raise SystemExit(f"이미 있습니다 (덮어쓰려면 --force): {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    for rel in FILES:
        src = run / rel
        if src.exists():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest / rel)

    info = provenance.read(best)
    (dest / provenance.RUN_INFO).write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
                                            encoding="utf-8")
    provenance.stamp_onnx(dest / "weights" / "best.onnx", info)

    git = info.get("git") or {}
    print(f"[archive] {run} → {dest}")
    print(f"[archive] 코드 {(git.get('commit') or '?')[:7]} ({git.get('branch')})  "
          f"dirty={git.get('dirty')}")
    for p in sorted(q for q in dest.rglob("*") if q.is_file()):
        print(f"  {p.relative_to(dest).as_posix():18s} {p.stat().st_size / 1e6:7.1f} MB")
    print("[archive] git add weights/ 후 커밋 — .pt · .onnx 는 Git LFS 로 올라간다 (.gitattributes)")


if __name__ == "__main__":
    main()
