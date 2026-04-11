#!/usr/bin/env python3
"""
Generate ground-truth low-level navigation paths for BEHAVIOR-1K scenes.

Reads scene_graph.json (room centroids, door positions, connectivity) and
traversability PNGs to produce A*-planned paths between all room pairs.
No OmniGibson / Isaac Sim required.

Outputs per scene:
  nav_paths.json            -- waypoint / trajectory data for every room pair
  nav_paths_floor_N.png     -- colored path overlay on the traversability map
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Patch, Rectangle
from PIL import Image

MAP_DEFAULT_RESOLUTION = 0.01
SNAP_SEARCH_RADIUS = 80  # pixels (~0.8 m at 0.01 m/px)
OVERLAY_CROP_MARGIN_PX = 25
CROP_TRAV_VS_NODE_AREA_RATIO = 6.0
GRAPH_NODE_CROP_MARGIN_PX = 60

DEFAULT_SCENES_ROOT = (
    Path(__file__).resolve().parent
    / "BEHAVIOR-1K"
    / "datasets"
    / "behavior-1k-assets"
    / "scenes"
)
DEFAULT_SELECTED = Path(__file__).resolve().parent / "selected_scenes.txt"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "scene_graphs"

TRAV_VARIANTS = [
    "floor_trav_no_obj_{idx}.png",
    "floor_trav_{idx}.png",
    "floor_trav_no_door_{idx}.png",
]

DOOR_PAINT_RADIUS_PX = 10  # half-width in pixels to paint around each door position
BRIDGE_PAINT_RADIUS_PX = 6  # half-width when bridging room_adjacent pairs with no door
BRIDGE_SCAN_STEPS = 100  # sample points along centroid-centroid line to find thinnest wall

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def world_xy_to_pixel(
    wx: float, wy: float, img_w: int, img_h: int, resolution: float
) -> tuple[int, int]:
    col = int(round(wx / resolution + img_w / 2.0))
    row = int(round(wy / resolution + img_h / 2.0))
    return col, row


def pixel_to_world_xy(
    col: int, row: int, img_w: int, img_h: int, resolution: float
) -> tuple[float, float]:
    wx = (col - img_w / 2.0) * resolution
    wy = (row - img_h / 2.0) * resolution
    return wx, wy


# ---------------------------------------------------------------------------
# Snap-to-free: BFS spiral to find nearest traversable pixel
# ---------------------------------------------------------------------------

def snap_to_free(grid: np.ndarray, row: int, col: int, radius: int = SNAP_SEARCH_RADIUS) -> tuple[int, int] | None:
    h, w = grid.shape
    if 0 <= row < h and 0 <= col < w and grid[row, col]:
        return row, col
    from collections import deque
    visited = set()
    q = deque()
    q.append((row, col))
    visited.add((row, col))
    while q:
        r, c = q.popleft()
        if 0 <= r < h and 0 <= c < w and grid[r, c]:
            return r, c
        if abs(r - row) > radius or abs(c - col) > radius:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return None


# ---------------------------------------------------------------------------
# A* on 8-connected binary grid
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_NEIGHBORS_8 = [
    (-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
    (0, -1, 1.0),                    (0, 1, 1.0),
    (1, -1, _SQRT2),  (1, 0, 1.0),  (1, 1, _SQRT2),
]


def astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """A* on a boolean grid. Returns pixel path [(row, col), ...] or None."""
    h, w = grid.shape
    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < h and 0 <= sc < w and grid[sr, sc]):
        return None
    if not (0 <= gr < h and 0 <= gc < w and grid[gr, gc]):
        return None

    def heuristic(r: int, c: int) -> float:
        dr, dc = abs(r - gr), abs(c - gc)
        return _SQRT2 * min(dr, dc) + abs(dr - dc)

    open_set: list[tuple[float, int, int, int]] = []  # (f, r, c, counter)
    counter = 0
    g_cost: dict[tuple[int, int], float] = {(sr, sc): 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    heapq.heappush(open_set, (heuristic(sr, sc), sr, sc, counter))

    while open_set:
        f, r, c, _ = heapq.heappop(open_set)

        if r == gr and c == gc:
            path = [(gr, gc)]
            cur = (gr, gc)
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        current_g = g_cost.get((r, c), float("inf"))
        if f - heuristic(r, c) > current_g + 1e-6:
            continue

        for dr, dc, cost in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not grid[nr, nc]:
                continue
            # Diagonal: both adjacent cardinal cells must be free to prevent corner-cutting
            if dr != 0 and dc != 0:
                if not grid[r + dr, c] or not grid[r, c + dc]:
                    continue
            ng = current_g + cost
            if ng < g_cost.get((nr, nc), float("inf")):
                g_cost[(nr, nc)] = ng
                counter += 1
                heapq.heappush(open_set, (ng + heuristic(nr, nc), nr, nc, counter))
                came_from[(nr, nc)] = (r, c)

    return None


# ---------------------------------------------------------------------------
# Waypoint downsampling (Ramer-Douglas-Peucker)
# ---------------------------------------------------------------------------

def _rdp_reduce(points: list[list[float]], epsilon: float) -> list[list[float]]:
    if len(points) <= 2:
        return points
    sx, sy = points[0]
    ex, ey = points[-1]
    dx, dy = ex - sx, ey - sy
    line_len = math.hypot(dx, dy)
    max_dist = 0.0
    max_idx = 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        if line_len < 1e-12:
            d = math.hypot(px - sx, py - sy)
        else:
            d = abs(dy * px - dx * py + ex * sy - ey * sx) / line_len
        if d > max_dist:
            max_dist = d
            max_idx = i
    if max_dist > epsilon:
        left = _rdp_reduce(points[: max_idx + 1], epsilon)
        right = _rdp_reduce(points[max_idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def downsample_waypoints(
    pixel_path: list[tuple[int, int]],
    img_w: int,
    img_h: int,
    resolution: float,
    spacing_m: float,
) -> list[list[float]]:
    """Convert pixel path to world XY and simplify with RDP + uniform spacing."""
    world = [
        list(pixel_to_world_xy(c, r, img_w, img_h, resolution))
        for r, c in pixel_path
    ]
    if len(world) <= 2:
        return world

    simplified = _rdp_reduce(world, epsilon=spacing_m * 0.4)

    resampled: list[list[float]] = [simplified[0]]
    accum = 0.0
    for i in range(1, len(simplified)):
        dx = simplified[i][0] - simplified[i - 1][0]
        dy = simplified[i][1] - simplified[i - 1][1]
        seg = math.hypot(dx, dy)
        accum += seg
        if accum >= spacing_m:
            resampled.append(simplified[i])
            accum = 0.0
    if resampled[-1] != simplified[-1]:
        resampled.append(simplified[-1])
    return resampled


# ---------------------------------------------------------------------------
# Scene graph loading
# ---------------------------------------------------------------------------

def load_scene_graph(json_path: Path) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def extract_rooms(graph_data: dict) -> list[dict]:
    return [
        n for n in graph_data["graph"]["nodes"] if n.get("type") == "room"
    ]


def extract_doors(graph_data: dict) -> list[dict]:
    return [
        n for n in graph_data["graph"]["nodes"] if n.get("type") == "door"
    ]


def extract_room_adjacency(graph_data: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for link in graph_data["graph"]["links"]:
        if link.get("relation") == "room_adjacent":
            pairs.append((link["source"], link["target"]))
    return pairs


def doors_between_rooms(doors: list[dict], room_a: str, room_b: str) -> list[str]:
    result = []
    for d in doors:
        cr = d.get("connected_rooms", [])
        if room_a in cr and room_b in cr:
            result.append(d["id"])
    return result


# ---------------------------------------------------------------------------
# Traversability map loading
# ---------------------------------------------------------------------------

def load_trav_map(
    layout_dir: Path, floor_idx: int, variant: str
) -> np.ndarray | None:
    pattern = variant.format(idx=floor_idx)
    p = layout_dir / pattern
    if p.is_file():
        img = Image.open(p).convert("L")
        return np.asarray(img) > 128
    return None


def find_trav_map(layout_dir: Path, floor_idx: int) -> tuple[np.ndarray, str] | None:
    for variant in TRAV_VARIANTS:
        grid = load_trav_map(layout_dir, floor_idx, variant)
        if grid is not None:
            return grid, variant.format(idx=floor_idx)
    return None


def paint_doors_on_grid(
    grid: np.ndarray,
    doors: list[dict],
    floor_idx: int,
    img_w: int,
    img_h: int,
    resolution: float,
    radius_px: int = DOOR_PAINT_RADIUS_PX,
) -> int:
    """Mark pixels around each door position as traversable. Returns count of doors painted."""
    h, w = grid.shape
    count = 0
    for d in doors:
        if int(d.get("floor_id", 0)) != floor_idx:
            continue
        pos = d.get("position")
        if not pos:
            continue
        col, row = world_xy_to_pixel(pos[0], pos[1], img_w, img_h, resolution)
        r0 = max(0, row - radius_px)
        r1 = min(h, row + radius_px + 1)
        c0 = max(0, col - radius_px)
        c1 = min(w, col + radius_px + 1)
        grid[r0:r1, c0:c1] = True
        count += 1
    return count


def connect_grid_components(
    grid: np.ndarray,
    rooms: list[dict],
    floor_idx: int,
    img_w: int,
    img_h: int,
    resolution: float,
    radius_px: int = BRIDGE_PAINT_RADIUS_PX,
) -> int:
    """Ensure all room centroids are on the same connected component.
    Uses scipy-free BFS labeling; bridges closest boundary pixels between components."""
    from collections import deque

    h, w = grid.shape
    floor_rooms = [
        r for r in rooms
        if int(r.get("floor_id", 0)) == floor_idx and r.get("centroid")
    ]
    if len(floor_rooms) < 2:
        return 0

    room_pixels: list[tuple[int, int, str]] = []
    for r in floor_rooms:
        c = r["centroid"]
        col, row = world_xy_to_pixel(c[0], c[1], img_w, img_h, resolution)
        snapped = snap_to_free(grid, row, col)
        if snapped:
            room_pixels.append((snapped[0], snapped[1], r["id"]))

    if len(room_pixels) < 2:
        return 0

    labels = np.full((h, w), -1, dtype=np.int32)
    comp_id = 0
    room_comp: dict[str, int] = {}

    for rr, rc, rid in room_pixels:
        if labels[rr, rc] >= 0:
            room_comp[rid] = int(labels[rr, rc])
            continue
        q = deque()
        q.append((rr, rc))
        labels[rr, rc] = comp_id
        while q:
            cr, cc = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] and labels[nr, nc] < 0:
                    labels[nr, nc] = comp_id
                    q.append((nr, nc))
        room_comp[rid] = comp_id
        comp_id += 1

    unique_comps = set(room_comp.values())
    if len(unique_comps) <= 1:
        return 0

    bridges = 0
    comp_list = sorted(unique_comps)
    merged: dict[int, int] = {c: c for c in comp_list}

    def find(x: int) -> int:
        while merged[x] != x:
            merged[x] = merged[merged[x]]
            x = merged[x]
        return x

    for i in range(len(comp_list)):
        for j in range(i + 1, len(comp_list)):
            ci, cj = find(comp_list[i]), find(comp_list[j])
            if ci == cj:
                continue
            rooms_i = [rp for rp in room_pixels if find(room_comp[rp[2]]) == ci]
            rooms_j = [rp for rp in room_pixels if find(room_comp[rp[2]]) == cj]
            best_dist = float("inf")
            best_mid = None
            for ri_r, ri_c, _ in rooms_i:
                for rj_r, rj_c, _ in rooms_j:
                    for step in range(BRIDGE_SCAN_STEPS + 1):
                        t = step / BRIDGE_SCAN_STEPS
                        mr = int(round(ri_r + t * (rj_r - ri_r)))
                        mc = int(round(ri_c + t * (rj_c - ri_c)))
                        if 0 <= mr < h and 0 <= mc < w and not grid[mr, mc]:
                            d = math.hypot(ri_r - rj_r, ri_c - rj_c)
                            if d < best_dist:
                                best_dist = d
                                best_mid = (mr, mc)
                                break
            if best_mid is None:
                mr = (rooms_i[0][0] + rooms_j[0][0]) // 2
                mc = (rooms_i[0][1] + rooms_j[0][1]) // 2
                best_mid = (mr, mc)

            pr, pc = best_mid
            pr0 = max(0, pr - radius_px)
            pr1 = min(h, pr + radius_px + 1)
            pc0 = max(0, pc - radius_px)
            pc1 = min(w, pc + radius_px + 1)
            grid[pr0:pr1, pc0:pc1] = True

            ri_r, ri_c = rooms_i[0][0], rooms_i[0][1]
            rj_r, rj_c = rooms_j[0][0], rooms_j[0][1]
            for step in range(BRIDGE_SCAN_STEPS + 1):
                t = step / BRIDGE_SCAN_STEPS
                sr = int(round(ri_r + t * (rj_r - ri_r)))
                sc = int(round(ri_c + t * (rj_c - ri_c)))
                if 0 <= sr < h and 0 <= sc < w and not grid[sr, sc]:
                    r0 = max(0, sr - radius_px)
                    r1 = min(h, sr + radius_px + 1)
                    c0_ = max(0, sc - radius_px)
                    c1_ = min(w, sc + radius_px + 1)
                    grid[r0:r1, c0_:c1_] = True

            merged[cj] = ci
            bridges += 1

    return bridges


def bridge_adjacent_rooms(
    grid: np.ndarray,
    graph_data: dict,
    rooms: list[dict],
    floor_idx: int,
    img_w: int,
    img_h: int,
    resolution: float,
    radius_px: int = BRIDGE_PAINT_RADIUS_PX,
) -> int:
    """For room_adjacent pairs with no connecting door, scan the line between
    centroids for the thinnest wall crossing and paint it traversable."""
    h, w = grid.shape
    room_map = {r["id"]: r for r in rooms if int(r.get("floor_id", 0)) == floor_idx}
    doors = [n for n in graph_data["graph"]["nodes"] if n.get("type") == "door"]
    count = 0

    for link in graph_data["graph"]["links"]:
        if link.get("relation") != "room_adjacent":
            continue
        ra_id, rb_id = link["source"], link["target"]
        if ra_id not in room_map or rb_id not in room_map:
            continue

        if doors_between_rooms(doors, ra_id, rb_id):
            continue

        ca = room_map[ra_id].get("centroid")
        cb = room_map[rb_id].get("centroid")
        if not ca or not cb:
            continue

        col_a, row_a = world_xy_to_pixel(ca[0], ca[1], img_w, img_h, resolution)
        col_b, row_b = world_xy_to_pixel(cb[0], cb[1], img_w, img_h, resolution)

        best_t = 0.5
        best_wall_width = float("inf")
        for i in range(BRIDGE_SCAN_STEPS + 1):
            t = i / BRIDGE_SCAN_STEPS
            cr = int(round(row_a + t * (row_b - row_a)))
            cc = int(round(col_a + t * (col_b - col_a)))
            if 0 <= cr < h and 0 <= cc < w and not grid[cr, cc]:
                wall_w = 0
                for dr in range(-radius_px * 2, radius_px * 2 + 1):
                    for dc in range(-radius_px * 2, radius_px * 2 + 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not grid[nr, nc]:
                            wall_w += 1
                if wall_w < best_wall_width:
                    best_wall_width = wall_w
                    best_t = t

        paint_r = int(round(row_a + best_t * (row_b - row_a)))
        paint_c = int(round(col_a + best_t * (col_b - col_a)))
        pr0 = max(0, paint_r - radius_px)
        pr1 = min(h, paint_r + radius_px + 1)
        pc0 = max(0, paint_c - radius_px)
        pc1 = min(w, paint_c + radius_px + 1)
        grid[pr0:pr1, pc0:pc1] = True
        count += 1

    return count


# ---------------------------------------------------------------------------
# Crop helpers (matching build_scene_graphs.py logic)
# ---------------------------------------------------------------------------

def trav_map_building_bbox(
    arr: np.ndarray, threshold: int = 12
) -> tuple[int, int, int, int] | None:
    if arr.ndim == 3:
        mask = np.any(arr > threshold, axis=2)
    else:
        mask = arr > threshold
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def node_bbox_from_rooms(
    rooms: list[dict],
    doors: list[dict],
    floor_idx: int,
    resolution: float,
    img_w: int,
    img_h: int,
    margin_px: int = GRAPH_NODE_CROP_MARGIN_PX,
) -> tuple[int, int, int, int] | None:
    rows_cols: list[tuple[int, int]] = []
    for r in rooms:
        if int(r.get("floor_id", 0)) != floor_idx:
            continue
        c = r.get("centroid")
        if c:
            col, row = world_xy_to_pixel(c[0], c[1], img_w, img_h, resolution)
            rows_cols.append((row, col))
    for d in doors:
        if int(d.get("floor_id", 0)) != floor_idx:
            continue
        p = d.get("position")
        if p:
            col, row = world_xy_to_pixel(p[0], p[1], img_w, img_h, resolution)
            rows_cols.append((row, col))
    if not rows_cols:
        return None
    rs = [rc[0] for rc in rows_cols]
    cs = [rc[1] for rc in rows_cols]
    return (
        max(0, min(rs) - margin_px),
        min(img_h, max(rs) + margin_px),
        max(0, min(cs) - margin_px),
        min(img_w, max(cs) + margin_px),
    )


def choose_crop_bbox(
    trav_crop: tuple[int, int, int, int],
    node_bbox: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    tr0, tr1, tc0, tc1 = trav_crop
    trav_area = max(0, tr1 - tr0) * max(0, tc1 - tc0)
    if node_bbox is None:
        return trav_crop
    nr0, nr1, nc0, nc1 = node_bbox
    node_area = max(0, nr1 - nr0) * max(0, nc1 - nc0)
    if node_area > 0 and trav_area > CROP_TRAV_VS_NODE_AREA_RATIO * node_area:
        return node_bbox
    return trav_crop


# ---------------------------------------------------------------------------
# Path generation for one scene
# ---------------------------------------------------------------------------

def generate_paths_for_scene(
    scene_name: str,
    scenes_root: Path,
    output_root: Path,
    waypoint_spacing: float,
) -> dict[str, Any] | None:
    out_dir = output_root / scene_name
    sg_path = out_dir / "scene_graph.json"
    if not sg_path.is_file():
        print(f"  [SKIP] {scene_name}: no scene_graph.json")
        return None

    graph_data = load_scene_graph(sg_path)
    rooms = extract_rooms(graph_data)
    doors = extract_doors(graph_data)

    layout_dir = scenes_root / scene_name / "layout"
    if not layout_dir.is_dir():
        print(f"  [SKIP] {scene_name}: no layout/ directory")
        return None

    num_floors = graph_data.get("num_floors", 1)
    resolution = MAP_DEFAULT_RESOLUTION

    all_paths: list[dict[str, Any]] = []
    total_ok = 0
    total_fail = 0

    for floor_idx in range(num_floors):
        result = find_trav_map(layout_dir, floor_idx)
        if result is None:
            print(f"  [SKIP] {scene_name} floor {floor_idx}: no trav map found")
            continue
        grid, trav_variant = result
        grid = grid.copy()  # writable copy for door painting
        img_h, img_w = grid.shape
        n_painted = paint_doors_on_grid(grid, doors, floor_idx, img_w, img_h, resolution)
        n_bridged = bridge_adjacent_rooms(grid, graph_data, rooms, floor_idx, img_w, img_h, resolution)
        n_comps = connect_grid_components(grid, rooms, floor_idx, img_w, img_h, resolution)
        print(f"  floor {floor_idx}: {trav_variant} ({img_w}x{img_h}), free={grid.sum()}, doors={n_painted}, adj_bridges={n_bridged}, comp_bridges={n_comps}")

        floor_rooms = [
            r for r in rooms
            if int(r.get("floor_id", 0)) == floor_idx and r.get("navigable", True)
        ]
        if len(floor_rooms) < 2:
            print(f"  [SKIP] floor {floor_idx}: fewer than 2 navigable rooms")
            continue

        room_map = {r["id"]: r for r in floor_rooms}

        for ra, rb in combinations(floor_rooms, 2):
            ra_id, rb_id = ra["id"], rb["id"]
            ca = ra.get("centroid")
            cb = rb.get("centroid")
            if not ca or not cb:
                total_fail += 1
                continue

            col_a, row_a = world_xy_to_pixel(ca[0], ca[1], img_w, img_h, resolution)
            col_b, row_b = world_xy_to_pixel(cb[0], cb[1], img_w, img_h, resolution)

            start = snap_to_free(grid, row_a, col_a)
            goal = snap_to_free(grid, row_b, col_b)
            if start is None or goal is None:
                total_fail += 1
                all_paths.append({
                    "path_id": f"{ra_id}__to__{rb_id}",
                    "floor_id": floor_idx,
                    "start": {"room": ra_id, "world_xy": [round(ca[0], 4), round(ca[1], 4)]},
                    "end": {"room": rb_id, "world_xy": [round(cb[0], 4), round(cb[1], 4)]},
                    "via_doors": doors_between_rooms(doors, ra_id, rb_id),
                    "status": "failed",
                    "failure_reason": "could not snap centroid to free pixel",
                    "path_length_m": None,
                    "num_waypoints": 0,
                    "waypoints": [],
                })
                continue

            pixel_path = astar(grid, start, goal)
            if pixel_path is None:
                total_fail += 1
                all_paths.append({
                    "path_id": f"{ra_id}__to__{rb_id}",
                    "floor_id": floor_idx,
                    "start": {"room": ra_id, "world_xy": [round(ca[0], 4), round(ca[1], 4)]},
                    "end": {"room": rb_id, "world_xy": [round(cb[0], 4), round(cb[1], 4)]},
                    "via_doors": doors_between_rooms(doors, ra_id, rb_id),
                    "status": "failed",
                    "failure_reason": "A* found no path",
                    "path_length_m": None,
                    "num_waypoints": 0,
                    "waypoints": [],
                })
                continue

            waypoints = downsample_waypoints(
                pixel_path, img_w, img_h, resolution, waypoint_spacing
            )
            path_length = 0.0
            for i in range(1, len(waypoints)):
                dx = waypoints[i][0] - waypoints[i - 1][0]
                dy = waypoints[i][1] - waypoints[i - 1][1]
                path_length += math.hypot(dx, dy)

            total_ok += 1
            all_paths.append({
                "path_id": f"{ra_id}__to__{rb_id}",
                "floor_id": floor_idx,
                "start": {"room": ra_id, "world_xy": [round(ca[0], 4), round(ca[1], 4)]},
                "end": {"room": rb_id, "world_xy": [round(cb[0], 4), round(cb[1], 4)]},
                "via_doors": doors_between_rooms(doors, ra_id, rb_id),
                "status": "ok",
                "path_length_m": round(path_length, 3),
                "num_waypoints": len(waypoints),
                "waypoints": [[round(x, 4), round(y, 4)] for x, y in waypoints],
            })

        draw_nav_paths_image(
            scene_name, floor_idx, grid, trav_variant,
            rooms, doors, all_paths, resolution, out_dir,
        )

    lengths = [p["path_length_m"] for p in all_paths if p["status"] == "ok"]
    envelope: dict[str, Any] = {
        "scene": scene_name,
        "generator": "build_nav_paths.py",
        "resolution_m_per_px": resolution,
        "waypoint_spacing_m": waypoint_spacing,
        "paths": all_paths,
        "summary": {
            "total_paths": total_ok + total_fail,
            "successful_paths": total_ok,
            "failed_paths": total_fail,
            "mean_path_length_m": round(sum(lengths) / len(lengths), 3) if lengths else None,
            "max_path_length_m": round(max(lengths), 3) if lengths else None,
            "min_path_length_m": round(min(lengths), 3) if lengths else None,
        },
    }

    out_json = out_dir / "nav_paths.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2)
    print(f"  wrote {out_json}  ({total_ok} ok, {total_fail} failed)")
    return envelope


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_COLORMAP = plt.colormaps["tab10"]


def _path_color(idx: int, total: int) -> tuple[float, ...]:
    return _COLORMAP(idx % 10)


def draw_nav_paths_image(
    scene_name: str,
    floor_idx: int,
    grid: np.ndarray,
    trav_variant: str,
    rooms: list[dict],
    doors: list[dict],
    all_paths: list[dict],
    resolution: float,
    out_dir: Path,
) -> None:
    img_h, img_w = grid.shape
    arr_u8 = (grid.astype(np.uint8) * 255)

    bbox = trav_map_building_bbox(arr_u8)
    margin = OVERLAY_CROP_MARGIN_PX
    if bbox is not None:
        r0, r1, c0, c1 = bbox
        r0 = max(0, r0 - margin)
        r1 = min(img_h, r1 + margin)
        c0 = max(0, c0 - margin)
        c1 = min(img_w, c1 + margin)
    else:
        r0, r1, c0, c1 = 0, img_h, 0, img_w

    trav_crop = (r0, r1, c0, c1)
    n_bbox = node_bbox_from_rooms(rooms, doors, floor_idx, resolution, img_w, img_h)
    r0, r1, c0, c1 = choose_crop_bbox(trav_crop, n_bbox)

    arr_rgb = np.stack([arr_u8[r0:r1, c0:c1]] * 3, axis=-1)
    cw, ch = c1 - c0, r1 - r0

    def to_px(wx: float, wy: float) -> tuple[float, float]:
        col, row = world_xy_to_pixel(wx, wy, img_w, img_h, resolution)
        return float(col - c0), float(row - r0)

    dpi = 150
    fig_w = max(4.0, cw / dpi)
    fig_h = max(3.0, ch / dpi)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(arr_rgb, origin="upper", aspect="equal")

    floor_paths = [p for p in all_paths if p["floor_id"] == floor_idx and p["status"] == "ok"]
    total_paths = len(floor_paths)

    legend_entries: list[Patch] = []
    for i, pdata in enumerate(floor_paths):
        wps = pdata["waypoints"]
        if len(wps) < 2:
            continue
        color = _path_color(i, total_paths)
        xs = []
        ys = []
        for wx, wy in wps:
            px, py = to_px(wx, wy)
            xs.append(px)
            ys.append(py)
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.85, solid_capstyle="round")
        ax.plot(xs[0], ys[0], "o", color=color, markersize=5, markeredgecolor="black", markeredgewidth=0.5)
        ax.plot(xs[-1], ys[-1], "s", color=color, markersize=5, markeredgecolor="black", markeredgewidth=0.5)
        label = (
            f"{pdata['start']['room']} \u2192 {pdata['end']['room']}"
            f"  ({pdata['path_length_m']:.1f}m)"
        )
        legend_entries.append(Patch(facecolor=color, edgecolor="k", label=label))

    fs = max(5.0, min(10.0, 800.0 / max(cw, ch, 1)))
    r_dot = max(4, min(10, int(max(cw, ch) / 80)))

    for room in rooms:
        if int(room.get("floor_id", 0)) != floor_idx:
            continue
        c = room.get("centroid")
        if not c:
            continue
        px, py = to_px(c[0], c[1])
        ax.add_patch(
            Circle(
                (px, py), r_dot,
                facecolor=to_rgba("#4682B4", 0.6),
                edgecolor="black", linewidth=0.8,
            )
        )
        ax.annotate(
            room["id"], xy=(px, py), xytext=(0, r_dot + 2),
            textcoords="offset points", fontsize=fs,
            ha="center", va="bottom", color="black",
            bbox={"boxstyle": "round,pad=0.1", "fc": "white", "alpha": 0.8},
        )

    for door in doors:
        if int(door.get("floor_id", 0)) != floor_idx:
            continue
        p = door.get("position")
        if not p:
            continue
        px, py = to_px(p[0], p[1])
        s = max(3, r_dot // 2)
        ax.add_patch(
            Rectangle(
                (px - s, py - s), 2 * s, 2 * s,
                facecolor="#FF8C00", edgecolor="black", linewidth=0.6,
            )
        )

    if legend_entries:
        ncol = 1 if len(legend_entries) <= 8 else 2
        legend_fs = max(4.0, min(7.0, 600.0 / max(cw, ch, 1)))
        ax.legend(
            handles=legend_entries, loc="upper left",
            fontsize=legend_fs, ncol=ncol,
            framealpha=0.85, borderpad=0.4, labelspacing=0.3,
        )

    ax.set_title(
        f"{scene_name}  |  navigation paths (floor {floor_idx})  [{total_paths} paths]",
        fontsize=fs + 1,
    )
    ax.set_axis_off()

    out_img = out_dir / f"nav_paths_floor_{floor_idx}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote {out_img}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth navigation paths for BEHAVIOR-1K scenes.",
    )
    parser.add_argument(
        "--scenes-root", type=Path, default=DEFAULT_SCENES_ROOT,
        help="Root directory containing scene subdirectories with layout/ and json/.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Output directory (scene_graph.json must already exist here per scene).",
    )
    parser.add_argument(
        "--scenes", nargs="*", default=None,
        help="Specific scene names to process (default: read selected_scenes.txt).",
    )
    parser.add_argument(
        "--waypoint-spacing", type=float, default=0.10,
        help="Target spacing between waypoints in meters (default: 0.10).",
    )
    args = parser.parse_args()

    if args.scenes:
        scene_names = args.scenes
    elif DEFAULT_SELECTED.is_file():
        scene_names = [
            ln.strip()
            for ln in DEFAULT_SELECTED.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    else:
        scene_names = sorted(
            d.name for d in args.scenes_root.iterdir()
            if d.is_dir() and (d / "layout").is_dir()
        )

    print(f"Processing {len(scene_names)} scene(s), waypoint spacing = {args.waypoint_spacing:.3f} m\n")

    results: list[dict[str, Any]] = []
    for scene_name in scene_names:
        print(f"=== {scene_name} ===")
        envelope = generate_paths_for_scene(
            scene_name, args.scenes_root, args.output_root, args.waypoint_spacing,
        )
        if envelope is not None:
            results.append(envelope)
        print()

    total_ok = sum(r["summary"]["successful_paths"] for r in results)
    total_fail = sum(r["summary"]["failed_paths"] for r in results)
    print(f"Done. {len(results)} scene(s) processed: {total_ok} paths ok, {total_fail} failed.")


if __name__ == "__main__":
    main()
