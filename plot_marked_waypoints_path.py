#!/usr/bin/env python3
"""
Render the robot path visiting the 3 red-X waypoints from
benevolence1_marked.pdf on top of the Benevolence_1_int trav map.

Steps:
  1) Detect the 3 red X centers in the marked PDF (rasterize via `sips`).
  2) Register the PDF to the native floor_trav_0.png (90-deg CCW + scale).
  3) Convert each X to world (x, y) using OmniGibson's world_to_map convention.
  4) Start the tour at the robot spawn used by autonomous_nav_benevolence1.py
     (X1 / living room: 2.02, 1.86), then visit the 3 Xs in visual (left->right);
     if START coincides with X1, skip duplicate X1 so planned segments are
     START→X2 and X2→X3.
     order. Snap each point to the nearest free cell on the eroded runtime map
     so paths are guaranteed A*-navigable for the robot.
  5) Run 8-connected A* on the eroded + wall-carved map between consecutive
     waypoints, and render a single annotated PNG with:
        - native trav map background
        - each path segment in a different color
        - waypoints (start, X1, X2, X3) labeled with world coords

Output:
  output/benevolence1_marked_path.png
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
NAT_PNG = (
    REPO
    / "BEHAVIOR-1K/datasets/behavior-1k-assets/scenes/Benevolence_1_int/layout/floor_trav_0.png"
)
DEFAULT_OUT_PNG = REPO / "output" / "benevolence1_marked_path.png"


def _legacy_cursor_pdf_path() -> Path:
    """Optional fallback for a machine-local Cursor workspace PDF (macOS only)."""
    return Path(
        "/Users/a65945/Library/Application Support/Cursor/User/workspaceStorage/"
        "a8272844e7f5cdff717c961ab76c11a7/pdfs/a655a807-5b13-4d59-94c9-cfeab3d99673/"
        "benevolence1_marked.pdf"
    )


def resolve_marked_pdf(pdf_arg: Path | None) -> Path | None:
    """
    Resolve the marked waypoint PDF: explicit --pdf, else repo docs/, else legacy Cursor path on macOS.
    Returns None if no file exists (caller prints error).
    """
    if pdf_arg is not None:
        p = pdf_arg.expanduser()
        return p if p.is_file() else None
    repo_pdf = REPO / "docs" / "benevolence1_marked.pdf"
    if repo_pdf.is_file():
        return repo_pdf
    if sys.platform == "darwin":
        leg = _legacy_cursor_pdf_path()
        if leg.is_file():
            return leg
    return None

# Match runtime: OmniGibson trav_map default is 0.01 m/px for the PNG; robot
# erosion radius (Turtlebot) ~0.30 m including +0.20 safety -> ~30 px.
NAT_RES_M = 0.01
EROSION_PX = 30

# Match autonomous_nav_benevolence1.py spawn (X1 living room).
START_WORLD = (2.02, 1.86)  # world (x, y)


def rasterize_pdf_hi(pdf: Path, out_png: Path, width_px: int = 2380) -> Path:
    """Use macOS `sips` to rasterize the PDF to a high-res PNG."""
    subprocess.run(
        [
            "sips",
            "-s",
            "format",
            "png",
            "-s",
            "formatOptions",
            "best",
            "--resampleWidth",
            str(width_px),
            str(pdf),
            "--out",
            str(out_png),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_png


def detect_red_xs(pdf_png: Path) -> list[tuple[int, int]]:
    """Return (px, py) centers of the 3 largest red marks, sorted by visual left->right."""
    im = cv2.imread(str(pdf_png), cv2.IMREAD_COLOR)
    b, g, r = cv2.split(im)
    red = r.astype(int) - np.maximum(g, b).astype(int)
    mask = (red > 60).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    num, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    for i in range(1, num):
        if 200 < stats[i, cv2.CC_STAT_AREA] < 20000:
            dets.append((int(cents[i, 0]), int(cents[i, 1])))
    dets.sort(key=lambda p: p[0])
    return dets


def pdf_to_native_transform(pdf_png: Path, nat_png: Path):
    """
    Return a function px->(world_x, world_y) along with useful intermediates.

    The marked PDF is the native trav map rotated 90 deg CCW, scaled, and
    placed inside a letter-size page. We register by matching bounding boxes
    of the binary free space in each image (rotation direction confirmed
    empirically via IoU ~0.96 for CCW vs 0.45 for CW).
    """
    pdf = cv2.imread(str(pdf_png), cv2.IMREAD_COLOR)
    b, g, r = cv2.split(pdf)
    red = r.astype(int) - np.maximum(g, b).astype(int)
    redmask = red > 60
    gray = cv2.cvtColor(pdf, cv2.COLOR_BGR2GRAY)
    gray[redmask] = 255
    pdf_bin = (gray > 200).astype(np.uint8)
    ys, xs = np.where(pdf_bin > 0)
    px0, py0, px1, py1 = xs.min(), ys.min(), xs.max(), ys.max()
    target_w = px1 - px0 + 1
    target_h = py1 - py0 + 1

    nat = cv2.imread(str(nat_png), cv2.IMREAD_GRAYSCALE)
    nat_bin = (nat > 200).astype(np.uint8)
    ys, xs = np.where(nat_bin > 0)
    nx0, ny0, nx1, ny1 = xs.min(), ys.min(), xs.max(), ys.max()
    nat_w = nx1 - nx0 + 1
    nat_h = ny1 - ny0 + 1
    nat_size = int(nat.shape[0])

    def px_to_world(px: int, py: int) -> tuple[float, float, float, float]:
        rx, ry = px - px0, py - py0
        r_orig = rx * nat_h / target_w
        c_orig = (nat_w - 1) - ry * nat_w / target_h
        full_row = ny0 + r_orig
        full_col = nx0 + c_orig
        wx = (full_col - nat_size / 2.0) * NAT_RES_M
        wy = (full_row - nat_size / 2.0) * NAT_RES_M
        return float(wx), float(wy), float(full_row), float(full_col)

    return px_to_world, nat, (nx0, ny0, nx1, ny1)


def world_to_pixel(wx: float, wy: float, nat_size: int = 1860):
    col = int(round(wx / NAT_RES_M + nat_size / 2.0))
    row = int(round(wy / NAT_RES_M + nat_size / 2.0))
    return row, col


def erode_free(nat_gray: np.ndarray, erosion_px: int) -> np.ndarray:
    """Binary free-map after eroding by robot footprint (robot fits if center cell is free)."""
    free = (nat_gray > 200).astype(np.uint8) * 255
    k = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), np.uint8)
    return cv2.erode(free, k)


def snap_to_nearest_free(free: np.ndarray, row: int, col: int, max_radius: int = 200):
    """BFS for the closest free cell; returns (row, col) or None."""
    h, w = free.shape
    if 0 <= row < h and 0 <= col < w and free[row, col] > 0:
        return row, col
    seen = np.zeros_like(free, dtype=bool)
    seen[row, col] = True
    dq = [(0, row, col)]
    while dq:
        d, r, c = dq.pop(0)
        if d > max_radius:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                continue
            seen[nr, nc] = True
            if free[nr, nc] > 0:
                return nr, nc
            dq.append((d + 1, nr, nc))
    return None


def load_nav_csv_xy(csv_path: Path) -> list[tuple[float, float]]:
    """Load (x, y) from autonomous_nav_benevolence1_nav.csv (or compatible)."""
    pts: list[tuple[float, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: need header row with x,y columns")
        for row in reader:
            pts.append((float(row["x"]), float(row["y"])))
    return pts


def astar(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """8-connected A* on a boolean free map. Returns list of (row, col) or None."""
    if free[start[0], start[1]] == 0 or free[goal[0], goal[1]] == 0:
        return None
    h, w = free.shape
    dxy = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    g = {start: 0.0}
    came = {}
    heap = [(0.0, 0.0, start)]
    while heap:
        f, cg, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return list(reversed(path))
        if cg > g.get(cur, math.inf):
            continue
        for dr, dc in dxy:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < h and 0 <= nc < w) or free[nr, nc] == 0:
                continue
            step = math.hypot(dr, dc)
            ng = cg + step
            if ng < g.get((nr, nc), math.inf):
                g[(nr, nc)] = ng
                came[(nr, nc)] = cur
                heur = math.hypot(nr - goal[0], nc - goal[1])
                heapq.heappush(heap, (ng + heur, ng, (nr, nc)))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot A* path + optional actual nav CSV overlay.")
    ap.add_argument(
        "--nav-csv",
        type=Path,
        default=REPO / "output" / "autonomous_nav_benevolence1_nav.csv",
        help="Robot trajectory CSV (x,y columns) for red overlay",
    )
    ap.add_argument(
        "--no-actual",
        action="store_true",
        help="Do not draw the actual path from the nav CSV",
    )
    ap.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Marked PDF with red X waypoints (default: docs/benevolence1_marked.pdf in repo, "
        "or legacy Cursor workspace path on macOS if present)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output PNG path (default: {DEFAULT_OUT_PNG})",
    )
    args = ap.parse_args()

    pdf_path = resolve_marked_pdf(args.pdf)
    if pdf_path is None:
        if args.pdf is not None:
            print(
                f"PDF not found: {args.pdf.expanduser()}\n"
                "Pass a valid --pdf path to benevolence1_marked.pdf.",
                file=sys.stderr,
            )
        else:
            print(
                "No marked PDF found. Place benevolence1_marked.pdf at:\n"
                f"  {REPO / 'docs' / 'benevolence1_marked.pdf'}\n"
                "or pass --pdf /path/to/benevolence1_marked.pdf",
                file=sys.stderr,
            )
        return 1

    out_png = args.out if args.out is not None else DEFAULT_OUT_PNG
    out_dir = out_png.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_hi = Path("/tmp/benevolence1_marked_hi.png")
    rasterize_pdf_hi(pdf_path, tmp_hi)

    pdf_xs = detect_red_xs(tmp_hi)
    if len(pdf_xs) != 3:
        print(f"[WARN] expected 3 red X marks, found {len(pdf_xs)}: {pdf_xs}")
    px_to_world, nat_gray, _ = pdf_to_native_transform(tmp_hi, NAT_PNG)
    worlds = [px_to_world(px, py) for (px, py) in pdf_xs]

    waypoints: list[tuple[str, float, float]] = [("START", *START_WORLD)]
    labels = ["X1", "X2", "X3"]
    for lbl, (wx, wy, _r, _c) in zip(labels, worlds):
        if (
            lbl == "X1"
            and math.hypot(wx - START_WORLD[0], wy - START_WORLD[1]) < 0.05
        ):
            continue
        waypoints.append((lbl, wx, wy))

    print("World waypoints:")
    for name, x, y in waypoints:
        print(f"  {name:5s} ({x:+.2f}, {y:+.2f}) m")

    # Eroded free map for planning
    free = erode_free(nat_gray, EROSION_PX)

    # Snap each waypoint to nearest free cell
    snapped_rc: list[tuple[int, int]] = []
    for name, x, y in waypoints:
        r, c = world_to_pixel(x, y, nat_gray.shape[0])
        s = snap_to_nearest_free(free, r, c)
        if s is None:
            print(f"[WARN] could not snap {name} ({x:+.2f},{y:+.2f})")
            snapped_rc.append((r, c))
        else:
            snapped_rc.append(s)
            if s != (r, c):
                sx = (s[1] - nat_gray.shape[0] / 2.0) * NAT_RES_M
                sy = (s[0] - nat_gray.shape[0] / 2.0) * NAT_RES_M
                print(
                    f"  {name}: snapped ({x:+.2f},{y:+.2f}) -> ({sx:+.2f},{sy:+.2f}) m"
                )

    # Plan A* between consecutive waypoints
    segments: list[list[tuple[int, int]]] = []
    for i in range(len(snapped_rc) - 1):
        path = astar(free, snapped_rc[i], snapped_rc[i + 1])
        if path is None:
            print(
                f"[WARN] A* FAILED: {waypoints[i][0]} -> {waypoints[i+1][0]}; drawing straight line"
            )
            path = [snapped_rc[i], snapped_rc[i + 1]]
        segments.append(path)
        dist_m = sum(
            math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
            for k in range(len(path) - 1)
        ) * NAT_RES_M
        print(
            f"  A* {waypoints[i][0]:>5s} -> {waypoints[i+1][0]:>5s} : {len(path)} cells, {dist_m:.2f} m"
        )

    # Render: BGR overlay on native trav map
    render = cv2.cvtColor(nat_gray, cv2.COLOR_GRAY2BGR)
    # Planned segments: default START→X2 green, X2→X3 "blue" (BGR); 3 segments if X1 kept
    if len(segments) == 3:
        seg_colors = [(0, 180, 255), (0, 200, 0), (255, 140, 0)]
    else:
        seg_colors = [(0, 200, 0), (255, 140, 0)]
    for col, seg in zip(seg_colors, segments):
        for k in range(len(seg) - 1):
            r0, c0 = seg[k]
            r1, c1 = seg[k + 1]
            cv2.line(render, (c0, r0), (c1, r1), col, thickness=5, lineType=cv2.LINE_AA)

    # Actual driven path (under waypoint circles)
    actual_pts: list[tuple[float, float]] = []
    robot_start_rc_full: tuple[int, int] | None = None
    if not args.no_actual and args.nav_csv.is_file():
        try:
            actual_pts = load_nav_csv_xy(args.nav_csv)
        except (OSError, ValueError) as e:
            print(f"[WARN] could not load nav CSV {args.nav_csv}: {e}")
    elif not args.no_actual:
        print(f"[WARN] nav CSV not found: {args.nav_csv} (skip actual overlay)")

    if actual_pts:
        nat_sz = nat_gray.shape[0]
        rc_list = [world_to_pixel(ax, ay, nat_sz) for ax, ay in actual_pts]
        robot_start_rc_full = rc_list[0]
        for k in range(len(rc_list) - 1):
            r0, c0 = rc_list[k]
            r1, c1 = rc_list[k + 1]
            cv2.line(render, (c0, r0), (c1, r1), (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        r0, c0 = rc_list[0]
        cv2.circle(render, (c0, r0), 10, (255, 255, 0), -1, lineType=cv2.LINE_AA)  # BGR cyan

    # Draw waypoints: START magenta, marked Xs red (circles only; labels added
    # after cropping/rotation so they land inside the image).
    n_x = len(waypoints) - 1
    wp_colors = [(255, 0, 255)] + [(0, 0, 230)] * n_x
    for rc, col in zip(snapped_rc, wp_colors):
        r, c = rc
        cv2.circle(render, (c, r), 18, (0, 0, 0), -1)
        cv2.circle(render, (c, r), 14, col, -1)

    # Crop tightly around the scene
    nx0, ny0, nx1, ny1 = (714, 808, 1252, 1837)
    pad = 60
    r0, r1 = max(0, ny0 - pad), min(render.shape[0], ny1 + pad)
    c0, c1 = max(0, nx0 - pad), min(render.shape[1], nx1 + pad)
    crop = render[r0:r1, c0:c1].copy()

    # Rotate 90 CCW so output matches the marked PDF view (+x right, +y down-right)
    rot = np.rot90(crop, k=1).copy()

    # Transform (row, col) in `crop` to (row, col) in `rot` (90 CCW):
    # Before rot: shape (H, W). After rot90 k=1: (W, H).
    # A point (r, c) -> (W - 1 - c, r)
    H_c, W_c = crop.shape[:2]

    def crop_rc_to_rot(rc):
        r, c = rc[0] - r0, rc[1] - c0
        return (W_c - 1 - c, r)

    # Add an on-canvas top border for title + legend after rotation
    H_r, W_r = rot.shape[:2]
    top = 200
    canvas = np.full((H_r + top, W_r, 3), 255, dtype=np.uint8)
    canvas[top:, :] = rot

    # Title
    cv2.putText(
        canvas,
        "Benevolence_1_int: A* path through 3 marked waypoints",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    # Legend (match segment count)
    if len(segments) == 3:
        legend = [
            ("Planned: START -> X1", seg_colors[0]),
            ("Planned: X1 -> X2", seg_colors[1]),
            ("Planned: X2 -> X3", seg_colors[2]),
        ]
    else:
        legend = [
            ("Planned: START -> X2", seg_colors[0]),
            ("Planned: X2 -> X3", seg_colors[1]),
        ]
    if actual_pts:
        legend.append(("Actual (CSV)", (0, 0, 255)))
    for i, (text, col) in enumerate(legend):
        y = 110 + 38 * i
        cv2.line(canvas, (30, y), (110, y), col, 8, cv2.LINE_AA)
        cv2.putText(canvas, text, (130, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    # Manual per-waypoint label offsets (pixels on the final canvas) to keep
    # nearby labels (e.g. START and X3, ~6 m apart) readable without overlap.
    label_offsets = {
        "START": (30, 42),
        "X1": (30, 0),
        "X2": (30, 28),
        "X3": (30, -22),
    }
    for (name, x, y), rc, col in zip(waypoints, snapped_rc, wp_colors):
        rc_rot = crop_rc_to_rot(rc)
        px, py = rc_rot[1], rc_rot[0] + top
        label = f"{name} ({x:+.2f},{y:+.2f})"
        (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        dx, dy = label_offsets.get(name, (30, 0))
        place_left = (px + dx + tw) > canvas.shape[1] - 10
        tx = px - 24 - tw if place_left else px + dx
        ty = py + 10 + dy
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2, cv2.LINE_AA)

    if robot_start_rc_full is not None:
        rc_rot_rs = crop_rc_to_rot(robot_start_rc_full)
        px_rs, py_rs = rc_rot_rs[1], rc_rot_rs[0] + top
        rs_label = "robot start"
        cv2.putText(
            canvas,
            rs_label,
            (px_rs + 14, py_rs + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 40, 40),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            rs_label,
            (px_rs + 14, py_rs + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_png), canvas)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
