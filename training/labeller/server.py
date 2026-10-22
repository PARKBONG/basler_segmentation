"""
raw/<source>/images/*.png 에 OBB 라벨(단일 클래스)을 붙이는 브라우저 라벨러.

카메라가 eye-in-hand 라 기구부(금속 튜브 + 와이어)는 화면에서 늘 같은 자리에 있고,
와이어는 튜브 출구를 축으로 휘거나 길이만 변합니다. 그래서 라벨러는 앱(BaslerLiveView)
과 같은 것을 보여주고, 와이어 박스를 "그리는" 대신 기준 자세에서 맞춰 옮깁니다:
  · 앱 config.xml <ReferenceOverlay> 의 누끼·윤곽선을 같은 불투명도로 겹쳐 표시
  · 와이어 박스 = 기준 와이어 OBB(reference_tube/wire 레이어, wire_template) 에서 시작해
    우측 상단 꼭짓점(튜브 출구, 피벗)을 기준으로 이동·회전 (+ 두께·길이)
  · 크롭 창을 앱처럼 W×H · 위치(%) 로 조절하고 preprocess.yaml 에 저장 (save_crop)
한 녹화 세션 안에서는 위치가 거의 같으므로, 1장 맞춰 "일괄 복사"로 그 세션의 나머지에
뿌린 뒤 어긋난 장만 고쳐 저장하는 흐름을 가정합니다 (복사는 지금 이미지의 세션 안에서만).

소스는 configs/preprocess.yaml 의 소스 이름입니다. 그 path 가 글롭(`raw/kimm_*`)이면
앱 REC 세션 폴더 전부를 이어서 보여줍니다 (Source 참고). 산출물은 preprocess.py 가
그대로 읽습니다 (이 소스가 이미 `type: obb` 이므로 회전 사각형 4점을 바로 씁니다):
    <세션 폴더>/labels/<이미지이름>.txt   "0 x1 y1 x2 y2 x3 y3 x4 y4" (정규화 4점)
    <세션 폴더>/data.yaml                  없으면 preprocess.yaml 의 최종 names 로 자동 생성
    <세션 폴더>/labels_seg/<이미지이름>.txt  와이어 폴리곤 "0 x1 y1 … xn yn" (compute edge 1 /
                                            OBB as edge 결과를 저장한 것 — labels/ 는 그대로)

실행 (저장소 루트에서 — uv 가 스크립트 위치로 training/pyproject.toml 을 찾습니다):
    uv run training/labeller/server.py [--source kimm] [--port 8765]
    브라우저에서 http://localhost:8765 접속.
    원격 GPU 머신에서 돌린다면 로컬에서: ssh -L 8765:localhost:8765 <머신>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import yaml
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
REPO = TRAINING.parent
DATASETS_RAW = REPO / "datasets" / "raw"
PREPROCESS_CFG = TRAINING / "configs" / "preprocess.yaml"
# 앱(BaslerLiveView) 설정 — 기구부 오버레이 파일·불투명도를 앱과 똑같이 쓰려고 읽는다
APP_CONFIG = REPO / "app" / "src" / "BaslerLiveView" / "config" / "config.xml"
# training/tools/make_reference_cutout.py --tube/--wire 의 입력 레이어 (풀 프레임 RGBA)
REF_TUBE = REPO / "datasets" / "reference" / "reference_tube.png"
REF_WIRE = REPO / "datasets" / "reference" / "reference_wire.png"

sys.path.insert(0, str(TRAINING))
from preprocess import min_area_rect  # noqa: E402 — training/ 를 path 에 넣은 뒤라야 import 됨
from edge import find_tip, obb_pose  # noqa: E402 — 같은 폴더 (labeller/)

IMG_EXTS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
           ".bmp": "image/bmp", ".webp": "image/webp"}
INDEX_HTML = HERE / "index.html"
CROP_KEYS = ("enabled", "width", "height", "center_x", "center_y")


# ── 기구부 오버레이 + 기준 와이어 ─────────────────────────────────────────────

def app_overlay_settings(config_xml: Path = APP_CONFIG) -> dict:
    """
    앱의 기구부 오버레이 설정 — config.xml 의 <ReferenceOverlay> 를 그대로 읽어 같은
    파일·같은 불투명도로 얹습니다 (값이 없거나 범위 밖이면 CameraConfig.cs 와 같은
    기본값). 상대경로는 저장소 루트 기준.
    """
    s = {"enabled": True, "path": "datasets/reference/reference_cutout.png", "opacity": 0.35,
         "outline_path": "datasets/reference/reference_outline.png", "outline_opacity": 0.9}
    node = ET.parse(config_xml).getroot().find("ReferenceOverlay") if config_xml.exists() else None
    if node is not None:
        def text(tag: str):
            v = node.findtext(tag)
            return v.strip() if v and v.strip() else None

        def opacity(tag: str, key: str) -> None:
            try:
                v = float(text(tag))
            except (TypeError, ValueError):
                return
            if 0 <= v <= 1:
                s[key] = v

        if text("Enabled") is not None:
            s["enabled"] = text("Enabled").lower() in ("true", "1")
        s["path"] = text("Path") or s["path"]
        s["outline_path"] = text("OutlinePath") or s["outline_path"]
        opacity("Opacity", "opacity")
        opacity("OutlineOpacity", "outline_opacity")

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else REPO / p

    return {"enabled": s["enabled"], "opacity": s["opacity"],
            "outline_opacity": s["outline_opacity"],
            "cutout": resolve(s["path"]), "outline": resolve(s["outline_path"])}


def wire_template(tube_path: Path = REF_TUBE, wire_path: Path = REF_WIRE) -> dict | None:
    """
    기준 와이어 OBB — 새 라벨의 초기값. 앱 오버레이의 빨간 와이어 박스와 같은 계산
    (training/tools/make_reference_cutout.py: 튜브 밖에 실제로 보이는 와이어 픽셀만의 회전 최소
    외접 사각형. 그쪽은 cv2 + 스무딩한 튜브 윤곽, 여기는 PIL + 튜브 알파 그대로라
    위 변이 몇 px 다를 수 있음). 가리는 튜브를 빼지 않으면 위 변이 튜브 속까지
    올라가 피벗이 튜브 출구에서 어긋납니다. 레이어가 없으면 None.
    """
    if not (tube_path.exists() and wire_path.exists()):
        return None

    def mask(p: Path):
        with Image.open(p) as im:
            return im.convert("RGBA").getchannel("A").point(lambda v: 255 if v else 0)

    wire, tube = mask(wire_path), mask(tube_path)
    if wire.size != tube.size:
        return None
    visible = ImageChops.subtract(wire, tube)
    w, h = visible.size
    pts = []
    for y in range(h):               # 행마다 양 끝 픽셀만 — 볼록 껍질에는 이것으로 충분
        bb = visible.crop((0, y, w, y + 1)).getbbox()
        if bb:
            pts += [(bb[0], y), (bb[2] - 1, y)]
    if len(pts) < 3:
        return None
    rect, area = min_area_rect(pts)
    if area <= 0:
        return None
    return {**obb_pose(rect), "size": [w, h]}


# ── preprocess.yaml 의 crop 제자리 수정 ───────────────────────────────────────

def save_crop(cfg_path: Path, source: str, values: dict) -> None:
    """
    preprocess.yaml 에서 sources[name=source].crop 의 enabled/width/height/center_x/
    center_y 값만 그 줄에서 고쳐 씁니다 — 주석·순서·다른 소스는 그대로 (yaml.safe_dump
    로 다시 쓰면 주석이 전부 날아갑니다). auto_crop 등 하위 블록은 건드리지 않고,
    없는 키는 crop: 바로 아래에 추가합니다.

    고친 텍스트를 다시 파싱해 "원래 내용 + 이 다섯 값"과 정확히 같을 때만 저장합니다
    (아니면 ValueError — 파일은 그대로).
    """
    want = {"enabled": bool(values["enabled"]),
            "width": int(values["width"]), "height": int(values["height"]),
            "center_x": float(values["center_x"]), "center_y": float(values["center_y"])}
    if want["width"] < 0 or want["height"] < 0:
        raise ValueError("width/height 는 0 이상이어야 합니다 (0 = 그 축 원본 전체)")
    for k in ("center_x", "center_y"):
        if not 0 <= want[k] <= 100:
            raise ValueError(f"{k} 는 0~100(%) 이어야 합니다: {want[k]}")

    def fmt(k: str) -> str:
        v = want[k]
        return ("true" if v else "false") if k == "enabled" else f"{v:g}"

    raw = cfg_path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.splitlines(keepends=True)

    def text(i: int) -> str:
        return lines[i].rstrip("\r\n")

    def lead(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    def is_code(s: str) -> bool:
        t = s.strip()
        return bool(t) and not t.startswith("#")

    item_re = re.compile(r"^(\s*)-\s+name:\s*['\"]?([^'\"#\s]+)['\"]?\s*(#.*)?$")
    start = dash = None
    for i in range(len(lines)):
        m = item_re.match(text(i))
        if m and m.group(2) == source:
            start, dash = i, len(m.group(1))
            break
    if start is None:
        raise ValueError(f"{cfg_path.name}: sources 에 {source} 가 없습니다")
    key_indent = text(start).index("name")       # 이 항목의 키들이 서는 열
    end = next((j for j in range(start + 1, len(lines))
                if is_code(text(j)) and lead(text(j)) <= dash), len(lines))

    crop_at = next((j for j in range(start + 1, end)
                    if is_code(text(j)) and lead(text(j)) == key_indent
                    and text(j).strip().startswith("crop:")), None)
    if crop_at is None:
        raise ValueError(f"{source}: crop: 블록이 없습니다 — preprocess.yaml 에 직접 추가하세요")
    if is_code(text(crop_at).strip()[len("crop:"):]):
        raise ValueError(f"{source}: crop 이 한 줄(flow) 형식이라 고칠 수 없습니다 — "
                         f"여러 줄 블록으로 바꿔 주세요")
    block_end = next((j for j in range(crop_at + 1, end)
                      if is_code(text(j)) and lead(text(j)) <= key_indent), end)
    children = [j for j in range(crop_at + 1, block_end) if is_code(text(j))]
    child_indent = min((lead(text(j)) for j in children), default=key_indent + 2)

    line_re = re.compile(r"^(\s*)(\w+):(\s*)([^#\r\n]*?)(\s*#[^\r\n]*)?(\r?\n)?$")
    found = set()
    for j in children:
        m = line_re.match(lines[j])
        if m and lead(text(j)) == child_indent and m.group(2) in CROP_KEYS:
            ind, key, sp, _, comment, nl = m.groups()
            lines[j] = f"{ind}{key}:{sp or ' '}{fmt(key)}{comment or ''}{nl or ''}"
            found.add(key)
    missing = [k for k in CROP_KEYS if k not in found]
    lines[crop_at + 1:crop_at + 1] = [f"{' ' * child_indent}{k}: {fmt(k)}{eol}" for k in missing]

    new = "".join(lines)
    expect = yaml.safe_load(raw) or {}
    for s in expect.get("sources") or []:
        if s.get("name") == source:
            if not isinstance(s.get("crop"), dict):
                s["crop"] = {}
            s["crop"].update(want)
    if (yaml.safe_load(new) or {}) != expect:
        raise ValueError(f"{cfg_path.name} 를 고친 결과가 의도와 달라 저장하지 않았습니다 — "
                         f"{source} 의 crop 블록을 직접 확인해 주세요")
    tmp = cfg_path.with_name(cfg_path.name + ".tmp")
    tmp.write_bytes(new.encode("utf-8"))
    os.replace(tmp, cfg_path)


class Source:
    """
    라벨링 대상 소스 — 경로 · 클래스 정의 · crop 미리보기 설정.

    `--source` 는 configs/preprocess.yaml 의 소스 **이름**입니다. 그 소스의 path 가
    글롭이면(앱 REC 는 녹화 세션마다 raw/kimm_YYMMDD_HHMMSS[_메모]/ 를 새로 만듭니다
    — FrameRecorder) 매칭되는 세션 폴더 전부를 한 소스처럼 이어서 라벨링합니다.
    preprocess.py 의 source_dirs 와 같은 규약이라, 여기서 라벨한 것이 그대로
    전처리 대상이 됩니다.

    라벨은 각 이미지가 있던 세션 폴더의 labels/ 에 씁니다 (preprocess.py 가 읽는 자리).
    앱 캡처 파일명(000001__cap_YYMMDD_HHMMSS.png)은 세션 타임스탬프가 들어가 전역으로
    유일하므로, 브라우저 API 는 파일명만 주고받고 소속 폴더는 서버가 기억합니다.

    preprocess.yaml 에 그 이름의 소스가 없으면 raw/<이름> 을 직접 씁니다 (글롭도 가능).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.cfg = (yaml.safe_load(PREPROCESS_CFG.read_text(encoding="utf-8")) or {}
                    if PREPROCESS_CFG.exists() else {})
        self.names = self._final_names()
        self.class_id = min(self.names) if self.names else 0
        self.dirs = self._resolve_dirs()

        # 이미지 파일명 → 그 이미지가 있는 세션 폴더
        self._owner: dict = {}
        for d in self.dirs:
            (d / "labels").mkdir(parents=True, exist_ok=True)
            self._ensure_data_yaml(d)
            for p in sorted((d / "images").iterdir()):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                prev = self._owner.get(p.name)
                if prev is not None:
                    # 파일명으로만 주고받는 구조라 겹치면 라벨이 엉뚱한 쪽에 붙습니다.
                    raise SystemExit(
                        f"파일명이 겹칩니다 — 한쪽을 지우거나 이름을 바꾸세요.\n"
                        f"  {prev / 'images' / p.name}\n  {p}")
                self._owner[p.name] = d

    def _source_entry(self) -> dict | None:
        """preprocess.yaml 의 sources 에서 이 이름의 소스 항목."""
        for src in self.cfg.get("sources") or []:
            if src.get("name") == self.name:
                return src
        return None

    def _resolve_dirs(self) -> list:
        """
        읽을 세션 폴더들 (이름순). preprocess.py 의 source_dirs 와 같은 규약 —
        글롭은 경로 마지막 칸에만, images/ 가 있는 폴더만 셉니다.
        """
        entry = self._source_entry() or {}
        raw = str(entry.get("path") or (DATASETS_RAW / self.name))
        base = Path(raw)
        base = base if base.is_absolute() else (TRAINING / base)

        if not any(c in base.name for c in "*?["):
            if not (base / "images").is_dir():
                raise SystemExit(f"이미지 폴더가 없습니다: {base / 'images'}")
            return [base]

        dirs = sorted(p for p in base.parent.glob(base.name)
                      if p.is_dir() and (p / "images").is_dir())
        if not dirs:
            raise SystemExit(f"'{base}' 에 맞는 폴더가 없습니다 "
                             f"(images/ 를 가진 폴더만 셉니다).")
        return dirs

    def _final_names(self) -> dict:
        """configs/preprocess.yaml 의 최종 클래스(names) — 없으면 기본 {0: wire}."""
        names = self.cfg.get("names")
        if names:
            return {int(k): str(v) for k, v in names.items()}
        return {0: "wire"}

    def _ensure_data_yaml(self, d: Path) -> None:
        """raw 소스 data.yaml 이 없으면 preprocess.py 가 요구하는 최소 형식으로 생성."""
        dy = d / "data.yaml"
        if dy.exists():
            return
        doc = {"names": self.names, "train": "images"}
        dy.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")

    def crop_state(self) -> dict | None:
        """
        configs/preprocess.yaml 의 이 소스 crop — 라벨러가 앱처럼 W×H · 위치(%) 로
        조절하고 save_crop() 으로 되돌려 씁니다. 창은 브라우저가 그림 — preprocess.py 의
        crop_rect() 와 같은 공식을 JS 로 한 번 더 계산합니다 (그쪽 공식이 바뀌면
        index.html 의 cropRect() 도 맞출 것). auto: 이미지마다 라벨에서 창을 잡는
        모드라 고정 창이 없음. preprocess.yaml 에 없는 소스면 None.
        """
        src = self._source_entry()
        if src is None:
            return None
        crop = src.get("crop") or {}
        return {"enabled": bool(crop.get("enabled")),
                "width": int(crop.get("width") or 0), "height": int(crop.get("height") or 0),
                "center_x": float(crop.get("center_x", 50)),
                "center_y": float(crop.get("center_y", 50)),
                "auto": bool((crop.get("auto_crop") or {}).get("enabled"))}

    def save_crop(self, values: dict) -> dict:
        """crop 을 preprocess.yaml 에 제자리 저장하고, 다시 읽은 상태를 돌려준다."""
        if self._source_entry() is None:
            raise ValueError(f"{PREPROCESS_CFG.name} 에 {self.name} 소스가 없어 크롭을 저장할 수 없습니다")
        save_crop(PREPROCESS_CFG, self.name, values)
        self.cfg = yaml.safe_load(PREPROCESS_CFG.read_text(encoding="utf-8")) or {}
        return self.crop_state()

    def images(self) -> list:
        # 세션 폴더 순 → 그 안에서 파일명 순 (_owner 의 삽입 순서). 파일명으로만
        # 정렬하면 세션들이 프레임 단위로 뒤섞여 "세션 하나 그리고 → 그 세션에 복사"
        # 흐름이 깨집니다.
        return list(self._owner)

    def dir_of(self, name: str) -> Path:
        d = self._owner.get(name)
        if d is None:
            raise ValueError(f"이 소스에 없는 이미지: {name!r}")
        return d

    def image_path(self, name: str) -> Path:
        return self.dir_of(name) / "images" / name

    def label_path(self, name: str) -> Path:
        return self.dir_of(name) / "labels" / (Path(name).stem + ".txt")

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
        # seg 폴리곤은 이 OBB 의 윗변·옆변으로 만든 것이라, OBB 가 바뀌면 낡는다 → 같이 지움
        self.seg_path(name).unlink(missing_ok=True)
        p = self.label_path(name)
        if not points:
            p.unlink(missing_ok=True)
            return
        flat = " ".join(f"{max(0.0, min(1.0, v)):.6f}" for xy in points for v in xy)
        p.write_text(f"{self.class_id} {flat}\n", encoding="utf-8")

    def delete_image(self, name: str) -> Path:
        """
        이미지 + 짝 라벨(labels/, labels_seg/)을 <raw>/_removed_manual/<세션>/ 으로 옮긴다 —
        글롭(kimm_*)에 안 걸려 라벨러·전처리에서 바로 빠지고, 잘못 지웠으면 되돌릴 수 있음.
        """
        d = self.dir_of(name)
        trash = d.parent / "_removed_manual" / d.name
        stem = Path(name).stem
        for sub, fname in (("images", name), ("labels", stem + ".txt"), ("labels_seg", stem + ".txt")):
            src = d / sub / fname
            if src.exists():
                (trash / sub).mkdir(parents=True, exist_ok=True)
                os.replace(src, trash / sub / fname)
        del self._owner[name]
        return trash

    def seg_path(self, name: str) -> Path:
        """seg 폴리곤 라벨 — 손으로 잡은 OBB(labels/) 와 따로 둔다."""
        return self.dir_of(name) / "labels_seg" / (Path(name).stem + ".txt")

    def write_seg(self, name: str, polygon: list) -> None:
        """폴리곤(이미지 px) → labels_seg/<이름>.txt (YOLO seg: 클래스 + 정규화 x y …)."""
        if len(polygon) < 3:
            raise ValueError(f"{name}: 폴리곤 점이 3개 미만입니다")
        with Image.open(self.image_path(name)) as im:   # 헤더만 읽음
            w, h = im.size
        flat = " ".join(f"{max(0.0, min(1.0, x / w)):.6f} {max(0.0, min(1.0, y / h)):.6f}"
                        for x, y in polygon)
        p = self.seg_path(name)
        p.parent.mkdir(exist_ok=True)
        p.write_text(f"{self.class_id} {flat}\n", encoding="utf-8")


def safe_name(raw: str) -> str:
    name = Path(raw).name  # 디렉터리 탈출 방지 — 파일명만 남긴다
    if not name or name != raw:
        raise ValueError(f"잘못된 파일명: {raw!r}")
    return name


def make_handler(source: Source, ref: dict | None = None):
    """
    ref = {"overlay": app_overlay_settings() 결과, "template": wire_template() 결과} —
    main 이 시작할 때 한 번 만들어 넘깁니다 (없으면 오버레이·기준 와이어 없이 동작).
    """
    ref = ref or {}
    overlay = ref.get("overlay")
    template = ref.get("template")

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

        def _file(self, fp: Path, content_type: str) -> None:
            if not fp.exists():
                self.send_error(404)
                return
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            try:
                if path == "/":
                    # 요청마다 읽는다 — index.html 을 고친 뒤 서버 재시작 없이 새로고침만으로 반영
                    self._file(INDEX_HTML, "text/html; charset=utf-8")
                elif path == "/api/config":
                    self._json({
                        "source": source.name,
                        "class_name": source.names.get(source.class_id, "?"),
                        "crop": source.crop_state(), "total": len(source.images()),
                        "dirs": len(source.dirs), "template": template,
                        "overlay": overlay and {
                            "enabled": overlay["enabled"], "opacity": overlay["opacity"],
                            "outline_opacity": overlay["outline_opacity"],
                            "cutout": overlay["cutout"].exists(),
                            "outline": overlay["outline"].exists()},
                    })
                elif path in ("/ref/cutout.png", "/ref/outline.png") and overlay:
                    self._file(overlay[path[len("/ref/"):-len(".png")]], "image/png")
                elif path == "/api/images":
                    self._json({"images": [{"name": n, "dir": source.dir_of(n).name,
                                            "labeled": source.label_path(n).exists(),
                                            "seg": source.seg_path(n).exists()}
                                           for n in source.images()]})
                elif path == "/api/label":
                    name = safe_name(query.get("name", [""])[0])
                    self._json({"points": source.read_label(name)})
                elif path == "/api/edge":
                    # 와이어 폴리곤 — method=1: 저장된 OBB 의 윗변·옆변 + 영상에서 찾은 끝
                    # (edge.find_tip, "compute edge 1"), method=obb: OBB 4점을 그대로
                    name = safe_name(query.get("name", [""])[0])
                    method = query.get("method", ["1"])[0]
                    points = source.read_label(name)
                    fp = source.image_path(name)
                    if not points:
                        self._json({"status": "no_label", "method": method})
                    elif method == "obb":
                        with Image.open(fp) as im:
                            w, h = im.size
                        self._json({"status": "ok", "method": "obb", "edges": [],
                                    "polygon": [[x * w, y * h] for x, y in points]})
                    elif method == "1":
                        self._json({**find_tip(fp, points), "method": "1"})
                    else:
                        raise ValueError(f"모르는 method: {method}")
                elif path.startswith("/image/"):
                    # 브라우저가 encodeURIComponent 로 보내므로 되돌린다 (공백 등)
                    name = safe_name(unquote(path[len("/image/"):]))
                    fp = source.image_path(name)
                    self._file(fp, IMG_EXTS[fp.suffix.lower()])
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
                    # 복사 범위는 지금 보고 있는 이미지의 세션 폴더 안으로 한정 —
                    # "타깃 위치가 거의 같다"는 전제는 한 녹화 세션 안에서만 성립합니다
                    # (세션마다 장면·지그가 다름). 소스가 폴더 하나면 기존과 같습니다.
                    current = safe_name(body.get("name", ""))
                    here = source.dir_of(current)
                    names = source.images()
                    # 범위 (목록 순서 = 세션 → 파일명). from_here: 지금 이미지부터 세션 끝까지,
                    # to_name: 지금 이미지와 그 이미지 사이(양 끝 포함, 순서 무관)
                    lo, hi = 0, len(names)
                    if body.get("to_name"):
                        i, j = names.index(current), names.index(safe_name(body["to_name"]))
                        lo, hi = min(i, j), max(i, j) + 1
                    elif body.get("from_here"):
                        lo = names.index(current)
                    written, skipped = [], 0
                    for n in names[lo:hi]:
                        if source.dir_of(n) != here:
                            continue
                        if not overwrite and source.label_path(n).exists():
                            skipped += 1
                            continue
                        source.write_label(n, points)
                        written.append(n)
                    # names: 실제로 OBB 가 바뀐 장 — 라벨러가 이 장들만 경계를 다시 계산
                    self._json({"ok": True, "written": len(written), "skipped": skipped,
                                "names": written})
                elif url.path == "/api/delete":
                    trash = source.delete_image(safe_name(self._body().get("name", "")))
                    self._json({"ok": True, "moved_to": str(trash)})
                elif url.path == "/api/seg":
                    # {"items": [{"name", "polygon": [[x, y], …] (이미지 px)}, …]} → labels_seg/
                    items = self._body().get("items") or []
                    for it in items:
                        source.write_seg(safe_name(it["name"]), it["polygon"])
                    self._json({"ok": True, "written": len(items)})
                elif url.path == "/api/crop":
                    self._json({"ok": True, "crop": source.save_crop(self._body())})
                else:
                    self.send_error(404)
            except (ValueError, KeyError, TypeError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="OBB 라벨러 (raw/<source>/images 대상)")
    ap.add_argument("--source", default="kimm",
                    help="configs/preprocess.yaml 의 소스 이름 (기본: kimm). 그 소스의 "
                         "path 가 글롭이면 매칭되는 세션 폴더 전부를 이어서 라벨링합니다. "
                         "preprocess.yaml 에 없는 이름이면 datasets/raw/<이름>/ 을 직접 씁니다.")
    ap.add_argument("--port", type=int, default=8765)
    # 인증이 없고 라벨 파일을 쓰는 서버라 기본은 로컬만 — 원격 머신이면 ssh -L 로 터널.
    ap.add_argument("--host", default="127.0.0.1", help="바인드 주소 (기본: 127.0.0.1)")
    args = ap.parse_args()

    source = Source(args.source)
    n = len(source.images())
    labeled = sum(1 for name in source.images() if source.label_path(name).exists())
    print(f"[labeller] source={source.name}  폴더={len(source.dirs)}개  images={n}  "
         f"labeled={labeled}  class={source.class_id}:{source.names.get(source.class_id)}")
    if len(source.dirs) > 1:
        print(f"[labeller]   {source.dirs[0].name} … {source.dirs[-1].name} "
             f"(라벨은 각 폴더의 labels/ 에 씁니다)")
    overlay = app_overlay_settings()
    template = wire_template()
    if template:
        print(f"[labeller] 기준 와이어: 피벗(우측 상단) "
             f"({template['pivot'][0]:.0f}, {template['pivot'][1]:.0f})  "
             f"축 {math.degrees(template['angle']):.1f}°  "
             f"길이 {template['length']:.0f}  두께 {template['thickness']:.0f} px")
    else:
        print(f"[labeller] 기준 와이어 없음 — {REF_TUBE.name}/{REF_WIRE.name} 가 없어 "
             f"새 라벨은 화면 가운데 기본 박스에서 시작합니다")
    print(f"[labeller] 오버레이: {overlay['cutout'].name} "
         f"{'있음' if overlay['cutout'].exists() else '없음'}, "
         f"{overlay['outline'].name} {'있음' if overlay['outline'].exists() else '없음'}")
    print(f"[labeller] http://localhost:{args.port}  (Ctrl+C 로 종료)")

    handler = make_handler(source, {"overlay": overlay, "template": template})
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
