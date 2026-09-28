"""datasets/reference.png 에서 기구부(금속 튜브 + 와이어) 누끼를 딴다.

카메라가 eye-in-hand 로 기구에 붙어 있어서, 화면 상단의 금속 튜브는 항상
같은 자리에 찍히고 와이어도 같은 자리에서 뻗어 나온다(길이만 변함).
그 영역을 알파 채널로 오려낸 RGBA PNG 를 reference.png 옆에 저장해 두면,
BaslerLiveView 가 시작할 때 읽어 라이브 뷰 위에 반투명으로 얹는다
(config\config.xml 의 <ReferenceOverlay> 참고).

분리 방법 — 배경은 아웃포커스로 흐릿한 중간 회색, 기구부는 그보다 어둡다:
  1) 큰 median blur 로 "기구부가 없다면 보였을" 배경 밝기를 픽셀별로 추정
  2) 추정 배경보다 일정 이상 어두운 픽셀(+ 아주 어두운 픽셀)을 후보로
  3) 화면 최상단에 닿아 있는 연결 성분만 남긴다 — 튜브는 위 프레임 밖에서
     들어오므로 반드시 상단에 닿고, 와이어는 튜브에 붙어 있다
  4) 경계를 살짝 feather 해서 알파로

사용:
  python tools/make_reference_cutout.py                # reference.png 에서 자동 분리
  python tools/make_reference_cutout.py --thresh 8     # 와이어가 덜 잡히면 낮추기

  # 자동 분리 대신, 그림판 등으로 직접 만든 RGBA 레이어(투명 배경 + 기구부)를
  # 합쳐서 누끼·윤곽선을 만들 수도 있다. 크기는 풀 프레임(2448×2048)이어야 한다:
  python tools/make_reference_cutout.py --layers datasets/reference_tube.png datasets/reference_wire.png

  # 튜브/와이어를 구분해서 주면 표시도 구분된다: 튜브는 초록 윤곽선,
  # 와이어는 빨간 OBB(회전 최소 외접 사각형). 누끼 채움은 둘을 합친 그대로:
  python tools/make_reference_cutout.py --tube datasets/reference_tube.png --wire datasets/reference_wire.png

출력:
  datasets/reference_cutout.png          — RGBA 누끼(앱이 읽는 파일)
  datasets/reference_outline.png         — 누끼 윤곽만 초록 선으로 그린 RGBA(앱이 같이 얹음)
  datasets/reference_cutout_preview.png  — 확인용(마젠타 배경에 합성 + 초록 윤곽)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def build_mask(gray: np.ndarray, thresh: int, dark: int) -> np.ndarray:
    """기구부(상단 연결 성분) 마스크를 uint8 0/255 로 반환."""
    # 1) 배경 추정: 튜브/와이어보다 훨씬 큰 커널의 median 은 기구부를 지우고
    #    배경의 완만한 명암만 남긴다.
    bg = cv2.medianBlur(gray, 151)

    # 2) 배경 대비 어두움 + 절대적으로 어두움(튜브 내부는 배경 추정이 튀어도
    #    확실히 잡히도록) 를 후보로.
    diff = bg.astype(np.int16) - gray.astype(np.int16)
    cand = ((diff > thresh) | (gray < dark)).astype(np.uint8) * 255

    # 3) 얇은 와이어가 중간에 끊기지 않게 닫고, 점 노이즈는 연다.
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k5, iterations=3)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k5)

    # 4) 최상단 행에 닿는 성분만 남긴다(튜브가 화면 위에서 들어오므로).
    n, labels = cv2.connectedComponents(cand, connectivity=8)
    top_labels = set(np.unique(labels[0:5, :])) - {0}
    mask = np.isin(labels, list(top_labels)).astype(np.uint8) * 255
    return mask


def smooth_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    """1차 스무딩: 마스크를 가우시안으로 뭉갠 뒤 절반에서 다시 이진화. 픽셀
    이진화가 만든 자잘한 톱니를 없애고 전체 크기·위치는 유지한다. 와이어
    (폭 ~수십 px)는 커널 σ보다 넓어 살아남는다."""
    if ksize < 3:
        return mask
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return ((blurred > 127).astype(np.uint8)) * 255


# 실루엣 스무딩·윤곽선 그리기 전에 마스크를 이만큼 복제 패딩한다. 마스크가
# 화면 위/왼쪽 프레임에 잘려 있어서, 그대로 윤곽을 찾으면 프레임을 따라가는
# 가짜 변이 생기고 스무딩도 프레임 모서리를 둥글린다. 패딩 위에서 작업한 뒤
# 가운데만 잘라내면 그 인공물이 전부 패딩 영역에 남고 프레임 안은 깨끗하다.
PAD = 64


def smooth_contours(mask: np.ndarray, sigma: float) -> list[np.ndarray]:
    """실루엣 윤곽을 호 길이 방향 가우시안으로 스무딩해 반환(패딩 좌표계).
    실제 금속 튜브 표면은 매끄러운데 이진화 실루엣에는 물방울 등 잔요철이
    남으므로, 닫힌 윤곽의 x(t), y(t)를 원형(wrap) 컨볼루션으로 저역 통과시켜
    σ보다 짧은 요철만 깎아낸다. 와이어 양쪽 변은 호 길이로는 멀리 떨어져
    있어 서로 뭉개지지 않고, 곧은 변·완만한 곡선은 그대로 유지된다."""
    padded = cv2.copyMakeBorder(mask, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if sigma <= 0:
        return [c.astype(np.int32) for c in contours]

    radius = max(1, int(round(sigma * 3)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()

    out = []
    for c in contours:
        pts = c[:, 0, :].astype(np.float64)
        if len(pts) <= 2 * radius:      # 커널보다 짧은 조각은 그대로
            out.append(c.astype(np.int32))
            continue
        sm = np.empty_like(pts)
        for d in range(2):              # 닫힌 곡선 → wrap 패딩 후 컨볼루션
            wrapped = np.concatenate([pts[-radius:, d], pts[:, d], pts[:radius, d]])
            sm[:, d] = np.convolve(wrapped, kernel, mode="valid")
        out.append(sm.round().astype(np.int32).reshape(-1, 1, 2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(REPO / "datasets" / "reference.png"))
    ap.add_argument("--out", default=str(REPO / "datasets" / "reference_cutout.png"))
    ap.add_argument("--thresh", type=int, default=10,
                    help="배경 대비 이만큼 어두우면 기구부 후보 (기본 10; 와이어가 끊기면 낮추기)")
    ap.add_argument("--dark", type=int, default=45,
                    help="이보다 어두운 픽셀은 무조건 후보 (기본 45)")
    ap.add_argument("--feather", type=int, default=3,
                    help="경계 feather 반경 px (기본 3)")
    ap.add_argument("--smooth", type=int, default=41,
                    help="1차 픽셀 스무딩 커널 px (기본 41; 0 = 없음)")
    ap.add_argument("--contour-sigma", type=float, default=20,
                    help="실루엣 호 길이 스무딩 σ px (기본 20; 0 = 없음)")
    ap.add_argument("--line", type=int, default=5,
                    help="윤곽선 두께 px (기본 5)")
    ap.add_argument("--layers", nargs="+", metavar="PNG",
                    help="자동 분리 대신 손으로 만든 RGBA 레이어들을 합쳐 누끼로 사용")
    ap.add_argument("--tube", metavar="PNG",
                    help="튜브 레이어(RGBA). --wire 와 함께 쓰며, 초록 윤곽선으로 표시")
    ap.add_argument("--wire", metavar="PNG",
                    help="와이어 레이어(RGBA). 빨간 OBB(회전 최소 외접 사각형)로 표시")
    args = ap.parse_args()

    if (args.tube is None) != (args.wire is None):
        ap.error("--tube 와 --wire 는 함께 주어야 합니다")
    if args.tube:
        args.layers = [args.tube, args.wire]

    tube_rough = wire_rough = None
    if args.layers:
        # 손으로 만든 레이어 모드: 각 RGBA 를 순서대로 알파 합성해 누끼 원본으로
        # 쓴다. 자동 이진화가 아니므로 1차 픽셀 스무딩은 건너뛴다(레이어의 알파가
        # 곧 의도한 모양). 실루엣 스무딩(--contour-sigma)은 똑같이 적용된다.
        bgr = alpha = None
        layer_masks = []
        for p in args.layers:
            layer = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if layer is None:
                raise SystemExit(f"레이어를 열 수 없음: {p}")
            if layer.ndim != 3 or layer.shape[2] != 4:
                raise SystemExit(f"알파 채널이 없음(RGBA 아님): {p}")
            if bgr is None:
                bgr = np.zeros(layer.shape[:2] + (3,), np.float32)
                alpha = np.zeros(layer.shape[:2], np.float32)
            elif layer.shape[:2] != alpha.shape:
                raise SystemExit(f"레이어 크기가 서로 다름: {p} {layer.shape[:2]} vs {alpha.shape}")
            fa = layer[:, :, 3].astype(np.float32) / 255
            bgr = layer[:, :, :3].astype(np.float32) * fa[:, :, None] + bgr * (1 - fa[:, :, None])
            alpha = fa + alpha * (1 - fa)
            layer_masks.append(((fa > 0.5).astype(np.uint8)) * 255)
        bgr = bgr.astype(np.uint8)
        rough = ((alpha > 0.5).astype(np.uint8)) * 255
        h, w = rough.shape
        if args.tube:
            tube_rough, wire_rough = layer_masks
        print(f"layers: {len(args.layers)} merged ({w}x{h})")
    else:
        bgr = cv2.imread(args.src, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"이미지를 열 수 없음: {args.src}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        rough = smooth_mask(build_mask(gray, args.thresh, args.dark), args.smooth)

    # 2차 스무딩: 실루엣 윤곽 자체를 매끄럽게 만들어 최종 마스크를 다시 채운다.
    # 누끼 알파와 초록 윤곽선이 같은 곡선에서 나오므로 둘이 정확히 겹친다.
    contours = smooth_contours(rough, args.contour_sigma)
    canvas = np.zeros((h + 2 * PAD, w + 2 * PAD), np.uint8)
    cv2.fillPoly(canvas, contours, 255)
    mask = canvas[PAD:PAD + h, PAD:PAD + w]
    coverage = (mask > 0).mean() * 100

    # feather: 마스크를 살짝 흐려 알파 경계를 부드럽게.
    ksize = args.feather * 2 + 1
    alpha = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    cv2.imwrite(args.out, rgba)

    # 초록 윤곽선만 담은 RGBA — 앱이 누끼와 별도 레이어로 얹는다(불투명도도 따로).
    # 패딩 캔버스에 그린 뒤 가운데를 잘라내므로, 마스크가 프레임에 잘린 곳에서
    # 프레임을 따라 도는 가짜 변은 패딩 영역에 그려졌다가 크롭으로 사라진다.
    outline_canvas = np.zeros((h + 2 * PAD, w + 2 * PAD, 4), np.uint8)
    green = (0, 200, 0, 255)  # BGRA
    red = (0, 0, 230, 255)
    if wire_rough is not None:
        # 튜브/와이어 구분 모드: 튜브는 실루엣을 초록으로, 와이어는 화소들의
        # 회전 최소 외접 사각형(OBB)을 빨강으로. 와이어는 길이가 변하는 부품이라
        # 정확한 실루엣보다 "이 영역 어딘가" 를 나타내는 박스가 알아보기 쉽다.
        tube_contours = smooth_contours(tube_rough, args.contour_sigma)
        cv2.drawContours(outline_canvas, tube_contours, -1, green, args.line, lineType=cv2.LINE_AA)
        pts = cv2.findNonZero(wire_rough)
        if pts is None:
            raise SystemExit(f"와이어 레이어가 비어 있음: {args.wire}")
        box = cv2.boxPoints(cv2.minAreaRect(pts)).round().astype(np.int32) + PAD
        cv2.polylines(outline_canvas, [box], True, red, args.line, lineType=cv2.LINE_AA)
    else:
        cv2.drawContours(outline_canvas, contours, -1, green, args.line, lineType=cv2.LINE_AA)
    outline = outline_canvas[PAD:PAD + h, PAD:PAD + w]
    outline_path = str(Path(args.out).with_name("reference_outline.png"))
    cv2.imwrite(outline_path, outline)

    # 확인용 프리뷰: 마젠타 배경 위에 알파 합성 + 초록 윤곽.
    magenta = np.zeros_like(bgr)
    magenta[:] = (255, 0, 255)
    a = (alpha.astype(np.float32) / 255)[:, :, None]
    preview = (bgr * a + magenta * (1 - a)).astype(np.uint8)
    oa = (outline[:, :, 3].astype(np.float32) / 255)[:, :, None]
    preview = (outline[:, :, :3] * oa + preview * (1 - oa)).astype(np.uint8)
    preview_path = str(Path(args.out).with_name(Path(args.out).stem + "_preview.png"))
    cv2.imwrite(preview_path, preview)

    print(f"mask coverage: {coverage:.1f}% of frame")
    print(f"cutout : {args.out}")
    print(f"outline: {outline_path}")
    print(f"preview: {preview_path}")


if __name__ == "__main__":
    main()
