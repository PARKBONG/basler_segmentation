"""
raw/<source>/images/*.png 에 OBB 라벨(단일 클래스)을 붙이는 브라우저 라벨러.

전제: 아직 완전한 데이터 획득 전이라 대부분의 장에서 타깃(wire) 위치가 거의 동일합니다.
그래서 1장만 정확히 그리고 "전체 이미지에 복사"로 나머지에 뿌린 뒤, 카메라가 흔들려
위치가 어긋난 장만 개별로 다시 그려 그 장만 저장하는 흐름을 가정합니다.

산출물은 preprocess.py 가 그대로 읽습니다 (configs/preprocess.yaml 의 이 소스가
이미 `type: obb` 이므로 폴리곤이 아니라 회전 사각형 4점을 바로 씁니다):
    raw/<source>/labels/<이미지이름>.txt   "0 x1 y1 x2 y2 x3 y3 x4 y4" (정규화 4점)
    raw/<source>/data.yaml                  없으면 preprocess.yaml 의 최종 names 로 자동 생성

실행:
    uv run python finetuner/labeller/server.py [--source kimm] [--port 8765]
    브라우저에서 http://localhost:8765 접속.
    원격 GPU 머신에서 돌린다면 로컬에서: ssh -L 8765:localhost:8765 <머신>
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import yaml

HERE = Path(__file__).resolve().parent
FINETUNER = HERE.parent
DATASETS_RAW = FINETUNER.parent / "datasets" / "raw"
PREPROCESS_CFG = FINETUNER / "configs" / "preprocess.yaml"

IMG_EXTS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
           ".bmp": "image/bmp", ".webp": "image/webp"}
INDEX_HTML = (HERE / "index.html").read_bytes()


class Source:
    """라벨링 대상 소스 (raw/<name>/) — 경로 · 클래스 정의 · crop 미리보기 설정."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = DATASETS_RAW / name
        self.images_dir = self.dir / "images"
        self.labels_dir = self.dir / "labels"
        if not self.images_dir.is_dir():
            raise SystemExit(f"이미지 폴더가 없습니다: {self.images_dir}")
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.names = self._final_names()
        self.class_id = min(self.names) if self.names else 0
        self._ensure_data_yaml()

    def _final_names(self) -> dict:
        """configs/preprocess.yaml 의 최종 클래스(names) — 없으면 기본 {0: wire}."""
        if PREPROCESS_CFG.exists():
            cfg = yaml.safe_load(PREPROCESS_CFG.read_text(encoding="utf-8")) or {}
            names = cfg.get("names")
            if names:
                return {int(k): str(v) for k, v in names.items()}
        return {0: "wire"}

    def _ensure_data_yaml(self) -> None:
        """raw 소스 data.yaml 이 없으면 preprocess.py 가 요구하는 최소 형식으로 생성."""
        dy = self.dir / "data.yaml"
        if dy.exists():
            return
        doc = {"names": self.names, "train": "images"}
        dy.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")

    def crop_preview(self) -> dict | None:
        """
        configs/preprocess.yaml 에서 이 소스의 crop 설정을 찾아 그대로 돌려준다
        (overlay 는 브라우저가 그림 — preprocess.py 의 crop_rect() 와 같은 공식을
        JS 로 한 번 더 계산합니다. 실제 크롭은 여전히 preprocess.py 가 전담하므로,
        그쪽 공식이 바뀌면 index.html 의 cropRect() 도 맞춰야 합니다).
        auto_crop 이 켜져 있으면(이미지마다 라벨에서 창을 잡음) 미리 보여줄 고정 창이
        없으므로 None.
        """
        if not PREPROCESS_CFG.exists():
            return None
        cfg = yaml.safe_load(PREPROCESS_CFG.read_text(encoding="utf-8")) or {}
        for src in cfg.get("sources", []):
            if src.get("name") != self.name:
                continue
            crop = src.get("crop") or {}
            if not crop.get("enabled") or (crop.get("auto_crop") or {}).get("enabled"):
                return None
            return {"width": crop.get("width", 0), "height": crop.get("height", 0),
                    "center_x": crop.get("center_x", 50), "center_y": crop.get("center_y", 50)}
        return None

    def images(self) -> list:
        return sorted(p.name for p in self.images_dir.iterdir()
                     if p.suffix.lower() in IMG_EXTS)

    def label_path(self, name: str) -> Path:
        return self.labels_dir / (Path(name).stem + ".txt")

    def read_label(self, name: str) -> list | None:
        p = self.label_path(name)
        if not p.exists():
            return None
        parts = p.read_text(encoding="utf-8").split()
        coords = [float(v) for v in parts[1:9]] if len(parts) >= 9 else None
        if not coords:
            return None
        return [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]

    def write_label(self, name: str, points: list | None) -> None:
        p = self.label_path(name)
        if not points:
            p.unlink(missing_ok=True)
            return
        flat = " ".join(f"{max(0.0, min(1.0, v)):.6f}" for xy in points for v in xy)
        p.write_text(f"{self.class_id} {flat}\n", encoding="utf-8")


def safe_name(raw: str) -> str:
    name = Path(raw).name  # 디렉터리 탈출 방지 — 파일명만 남긴다
    if not name or name != raw:
        raise ValueError(f"잘못된 파일명: {raw!r}")
    return name


def make_handler(source: Source):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:
            pass  # 조용히 — 필요하면 print(fmt % args)

        def _json(self, obj, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            try:
                if path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(INDEX_HTML)))
                    self.end_headers()
                    self.wfile.write(INDEX_HTML)
                elif path == "/api/config":
                    self._json({"source": source.name, "class_name": source.names.get(source.class_id, "?"),
                               "crop": source.crop_preview(), "total": len(source.images())})
                elif path == "/api/images":
                    self._json({"images": [{"name": n, "labeled": source.label_path(n).exists()}
                                           for n in source.images()]})
                elif path == "/api/label":
                    name = safe_name(query.get("name", [""])[0])
                    self._json({"points": source.read_label(name)})
                elif path.startswith("/image/"):
                    # 브라우저가 encodeURIComponent 로 보내므로 되돌린다 (공백 등)
                    name = safe_name(unquote(path[len("/image/"):]))
                    fp = source.images_dir / name
                    if not fp.exists():
                        self.send_error(404)
                        return
                    data = fp.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", IMG_EXTS[fp.suffix.lower()])
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_error(404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            try:
                if url.path == "/api/label":
                    body = self._body()
                    name = safe_name(body["name"])
                    source.write_label(name, body.get("points"))
                    self._json({"ok": True, "labeled": sum(
                        1 for n in source.images() if source.label_path(n).exists())})
                elif url.path == "/api/copy":
                    body = self._body()
                    points = body.get("points")
                    if not points:
                        raise ValueError("points 가 없습니다 — 복사할 라벨이 없습니다.")
                    overwrite = bool(body.get("overwrite"))
                    written = skipped = 0
                    for n in source.images():
                        if not overwrite and source.label_path(n).exists():
                            skipped += 1
                            continue
                        source.write_label(n, points)
                        written += 1
                    self._json({"ok": True, "written": written, "skipped": skipped})
                else:
                    self.send_error(404)
            except (ValueError, KeyError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="OBB 라벨러 (raw/<source>/images 대상)")
    ap.add_argument("--source", default="kimm", help="datasets/raw/<이름>/ (기본: kimm)")
    ap.add_argument("--port", type=int, default=8765)
    # 인증이 없고 라벨 파일을 쓰는 서버라 기본은 로컬만 — 원격 머신이면 ssh -L 로 터널.
    ap.add_argument("--host", default="127.0.0.1", help="바인드 주소 (기본: 127.0.0.1)")
    args = ap.parse_args()

    source = Source(args.source)
    n = len(source.images())
    labeled = sum(1 for name in source.images() if source.label_path(name).exists())
    print(f"[labeller] source={source.name}  images={n}  labeled={labeled}  "
         f"class={source.class_id}:{source.names.get(source.class_id)}")
    print(f"[labeller] http://localhost:{args.port}  (Ctrl+C 로 종료)")

    ThreadingHTTPServer((args.host, args.port), make_handler(source)).serve_forever()


if __name__ == "__main__":
    main()
