#!/usr/bin/env python3
"""
Render the planned (and optionally actual) robot path through the 7 marked
waypoints x1..x7 on the Benevolence_2_int traversability map.

Mirrors plot_marked_waypoints_path.py (which targets Benevolence_1_int / 3 Xs):
  1) Hard-code the 7 (room, x, y) waypoints in world meters. Coordinates come
     from output/scene_graphs/Benevolence_2_int/nav_paths.json (bathroom_0,
     bedroom_0, closet_0, corridor_0, closet_1, bedroom_1, bathroom_1) and
     match the user-marked image. Spawn = x1.
  2) For each consecutive pair, look up the room-to-room path that
     build_nav_paths.py already computed (it carves door openings, so the
     resulting polyline is the actual A* solution the runtime sim uses).
     Reverse the polyline if the alphabetical path goes the other way.
  3) Render onto floor_trav_0.png with one color per planned segment, plus a
     red overlay of the actual driven trajectory if a nav CSV is supplied.

Output:
  output/benevolence2_marked_path.png

The dataset's floor_trav_0.png lives in BEHAVIOR-1K/datasets/... which is
gitignored locally; pass --trav PATH or stage it under .cache/ first
(see run_benevolence2_ec2_and_pull.sh). Default search order:
  1. --trav PATH
  2. BEHAVIOR-1K/datasets/behavior-1k-assets/scenes/Benevolence_2_int/layout/floor_trav_0.png
  3. .cache/Benevolence_2_int/floor_trav_0.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
DATASET_TRAV = (
    REPO
    / "BEHAVIOR-1K/datasets/behavior-1k-assets/scenes/Benevolence_2_int/layout/floor_trav_0.png"
)
CACHE_TRAV = REPO / ".cache/Benevolence_2_int/floor_trav_0.png"
NAV_PATHS_JSON = REPO / "output" / "scene_graphs" / "Benevolence_2_int" / "nav_paths.json"
OUT_PNG = REPO / "output" / "benevolence2_marked_path.png"

NAT_RES_M = 0.01

# x4/x5 adjusted; x6 bedroom_1 centroid south of bed (matches nav_paths closet_1<->bedroom_1).
WAYPOINTS_WORLD: list[tuple[str, float, float]] = [
    ("X1", 0.4523, 6.0895),
    ("X2", 3.0923, 5.602),
    ("X3", 3.0923, 3.737),
    ("X4", 0.52, 2.33),
    ("X5", 0.10, -2.15),
    ("X6", 2.85, -1.06),
    ("X7", 3.0923, 2.2295),
]

ROOMS = {
    "X1": "bathroom_0",
    "X2": "bedroom_0",
    "X3": "closet_0",
    "X4": "corridor_0",
    "X5": "closet_1",
    "X6": "bedroom_1",
    "X7": "bathroom_1",
}


def world_to_pixel(wx: float, wy: float, nat_size: int) -> tuple[int, int]:
    col = int(round(wx / NAT_RES_M + nat_size / 2.0))
    row = int(round(wy / NAT_RES_M + nat_size / 2.0))
    return row, col


def pixel_to_world(row: int, col: int, nat_size: int) -> tuple[float, float]:
    wx = (col - nat_size / 2.0) * NAT_RES_M
    wy = (row - nat_size / 2.0) * NAT_RES_M
    return wx, wy


def load_nav_paths(json_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Return mapping path_id -> list of (x,y) waypoints in meters."""
    with open(json_path, encoding="utf-8") as f:
        blob = json.load(f)
    out: dict[str, list[tuple[float, float]]] = {}
    for entry in blob.get("paths", []):
        pid = entry.get("path_id")
        wps = entry.get("waypoints") or []
        if not pid or not wps:
            continue
        out[pid] = [(float(p[0]), float(p[1])) for p in wps]
    return out


def get_room_path_xy(
    paths_by_id: dict[str, list[tuple[float, float]]],
    room_a: str,
    room_b: str,
) -> tuple[list[tuple[float, float]], bool]:
    """
    Return (waypoints from room_a -> room_b, ok). Falls back to a straight
    line between the two endpoints if no path exists.
    """
    fwd = f"{room_a}__to__{room_b}"
    rev = f"{room_b}__to__{room_a}"
    if fwd in paths_by_id:
        return paths_by_id[fwd], True
    if rev in paths_by_id:
        return list(reversed(paths_by_id[rev])), True
    return [], False


def load_nav_csv_xy(csv_path: Path) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: need header row with x,y columns")
        for row in reader:
            pts.append((float(row["x"]), float(row["y"])))
    return pts


def find_trav_png(arg_path: Path | None) -> Path:
    if arg_path is not None:
        if not arg_path.is_file():
            raise FileNotFoundError(f"--trav {arg_path}: not a file")
        return arg_path
    for cand in (DATASET_TRAV, CACHE_TRAV):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "floor_trav_0.png not found. Pass --trav PATH or stage the file at "
        f"{CACHE_TRAV} (e.g. rsync from EC2)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot planned A* path through 7 marked Benevolence_2_int waypoints (and optional actual trajectory)."
    )
    ap.add_argument("--trav", type=Path, default=None, help="Path to floor_trav_0.png")
    ap.add_argument(
        "--nav-paths",
        type=Path,
        default=NAV_PATHS_JSON,
        help="Path to nav_paths.json from build_nav_paths.py (room-to-room polylines).",
    )
    ap.add_argument(
        "--nav-csv",
        type=Path,
        default=REPO / "output" / "autonomous_nav_benevolence2_nav.csv",
        help="Robot trajectory CSV (x,y columns) for red overlay (optional).",
    )
    ap.add_argument(
        "--no-actual",
        action="store_true",
        help="Skip the red actual-trajectory overlay even if the CSV exists.",
    )
    ap.add_argument("--out", type=Path, default=OUT_PNG, help="Output PNG path.")
    args = ap.parse_args()

    trav_path = find_trav_png(args.trav)
    nat_gray = cv2.imread(str(trav_path), cv2.IMREAD_GRAYSCALE)
    if nat_gray is None:
        print(f"[FATAL] could not read {trav_path}", file=sys.stderr)
        return 1
    nat_h, nat_w = nat_gray.shape[:2]
    if nat_h != nat_w:
        print(f"[WARN] non-square trav map {nat_w}x{nat_h}; world_to_pixel assumes square", file=sys.stderr)
    nat_size = nat_h

    print("World waypoints (x4/x5 adjusted; x6 south of bed; rest match nav_paths centroids):")
    for name, x, y in WAYPOINTS_WORLD:
        print(f"  {name} {ROOMS.get(name, '?'):>11s}  ({x:+.2f}, {y:+.2f}) m")

    if not args.nav_paths.is_file():
        print(f"[FATAL] {args.nav_paths} not found; run build_nav_paths.py first.", file=sys.stderr)
        return 1
    paths_by_id = load_nav_paths(args.nav_paths)

    snapped_rc: list[tuple[int, int]] = [
        world_to_pixel(x, y, nat_size) for _, x, y in WAYPOINTS_WORLD
    ]

    # Build per-segment polyline in pixel space using the precomputed nav_paths.
    segments: list[list[tuple[int, int]]] = []
    for i in range(len(WAYPOINTS_WORLD) - 1):
        a_name, _, _ = WAYPOINTS_WORLD[i]
        b_name, _, _ = WAYPOINTS_WORLD[i + 1]
        room_a = ROOMS[a_name]
        room_b = ROOMS[b_name]
        wps_xy, ok = get_room_path_xy(paths_by_id, room_a, room_b)
        if not ok:
            print(
                f"[WARN] no nav_paths entry for {room_a} -> {room_b}; using straight line"
            )
            wps_rc = [snapped_rc[i], snapped_rc[i + 1]]
        else:
            wps_rc = [world_to_pixel(x, y, nat_size) for (x, y) in wps_xy]
            # Anchor first/last cells to the actual waypoint pixels so segments connect cleanly.
            if wps_rc:
                wps_rc[0] = snapped_rc[i]
                wps_rc[-1] = snapped_rc[i + 1]
        segments.append(wps_rc)
        dist_m = (
            sum(
                math.hypot(wps_rc[k + 1][0] - wps_rc[k][0], wps_rc[k + 1][1] - wps_rc[k][1])
                for k in range(len(wps_rc) - 1)
            )
            * NAT_RES_M
        )
        print(
            f"  Path {a_name} -> {b_name} ({room_a:>11s} -> {room_b:<11s}): "
            f"{len(wps_rc):3d} pts, {dist_m:5.2f} m"
        )

    # Render
    render = cv2.cvtColor(nat_gray, cv2.COLOR_GRAY2BGR)

    # 6 distinct BGR colors for X1->X2, X2->X3, ..., X6->X7.
    seg_colors = [
        (0, 180, 255),    # orange
        (0, 200, 0),      # green
        (255, 140, 0),    # azure-blue
        (200, 0, 200),    # magenta
        (0, 220, 220),    # yellow
        (255, 60, 60),    # blue-cyan
    ]
    for col, seg in zip(seg_colors, segments):
        for k in range(len(seg) - 1):
            r0, c0 = seg[k]
            r1, c1 = seg[k + 1]
            cv2.line(render, (c0, r0), (c1, r1), col, thickness=5, lineType=cv2.LINE_AA)

    # Optional actual trajectory (red).
    actual_pts: list[tuple[float, float]] = []
    if not args.no_actual and args.nav_csv.is_file():
        try:
            actual_pts = load_nav_csv_xy(args.nav_csv)
        except (OSError, ValueError) as e:
            print(f"[WARN] could not load nav CSV {args.nav_csv}: {e}")
    elif not args.no_actual:
        print(f"[INFO] nav CSV not found at {args.nav_csv}; skipping actual overlay")

    if actual_pts:
        rc_list = [world_to_pixel(ax, ay, nat_size) for ax, ay in actual_pts]
        for k in range(len(rc_list) - 1):
            r0, c0 = rc_list[k]
            r1, c1 = rc_list[k + 1]
            cv2.line(render, (c0, r0), (c1, r1), (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        rs = rc_list[0]
        cv2.circle(render, (rs[1], rs[0]), 10, (255, 255, 0), -1, lineType=cv2.LINE_AA)

    # Waypoint markers (red on black halo).
    for rc in snapped_rc:
        r, c = rc
        cv2.circle(render, (c, r), 18, (0, 0, 0), -1)
        cv2.circle(render, (c, r), 14, (0, 0, 230), -1)

    # Auto-crop to the inhabited region with padding.
    free_orig = (nat_gray > 200).astype(np.uint8)
    ys, xs = np.where(free_orig > 0)
    if ys.size == 0:
        crop = render
        r0_off, c0_off = 0, 0
    else:
        ny0, ny1 = int(ys.min()), int(ys.max())
        nx0, nx1 = int(xs.min()), int(xs.max())
        pad = 60
        r0_off = max(0, ny0 - pad)
        r1 = min(render.shape[0], ny1 + pad)
        c0_off = max(0, nx0 - pad)
        c1 = min(render.shape[1], nx1 + pad)
        crop = render[r0_off:r1, c0_off:c1].copy()

    H_c, W_c = crop.shape[:2]
    top = 230
    canvas = np.full((H_c + top, max(W_c, 720), 3), 255, dtype=np.uint8)
    canvas[top:top + H_c, :W_c] = crop

    cv2.putText(
        canvas,
        "Benevolence_2_int: planned A* path through 7 marked waypoints (x1 -> x7)",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    legend: list[tuple[str, tuple[int, int, int]]] = []
    for i, color in enumerate(seg_colors):
        a, b = WAYPOINTS_WORLD[i][0], WAYPOINTS_WORLD[i + 1][0]
        legend.append((f"Planned: {a} -> {b}", color))
    if actual_pts:
        legend.append(("Actual (CSV)", (0, 0, 255)))

    cols = 2
    col_w = max(W_c, 720) // cols
    for i, (text, col) in enumerate(legend):
        col_idx = i % cols
        row_idx = i // cols
        x0 = 30 + col_idx * col_w
        y = 90 + 30 * row_idx
        cv2.line(canvas, (x0, y), (x0 + 60, y), col, 6, cv2.LINE_AA)
        cv2.putText(canvas, text, (x0 + 75, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    # Waypoint labels on the cropped canvas (transform full-image (r, c) -> canvas (px, py)).
    label_offset_px = {
        "X1": (16, -10),
        "X2": (16, 6),
        "X3": (16, 6),
        "X4": (-110, 6),
        "X5": (-110, 6),
        "X6": (16, 6),
        "X7": (16, 6),
    }
    for (name, wx, wy), (r, c) in zip(WAYPOINTS_WORLD, snapped_rc):
        px = c - c0_off
        py = (r - r0_off) + top
        label = f"{name} {ROOMS.get(name, '')} ({wx:+.2f},{wy:+.2f})"
        dx, dy = label_offset_px.get(name, (16, 6))
        cv2.putText(canvas, label, (px + dx, py + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + dx, py + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 200), 1, cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
