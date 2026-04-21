#!/usr/bin/env python3
"""
First-person robot navigation videos for BEHAVIOR-1K scenes.

Uses room centroids from output/scene_graphs/<scene>/nav_paths.json in a **receding-horizon**
tour (farthest unvisited room next). Navigation matches ``autonomous_nav_60s.py``: live A-star
on the sim traversability map (with objects + painted doorways), **pure pursuit** on the
replanned polyline, **LiDAR avoidance** (blend), **bearing-to-goal** when A* has no path
(with forward motion suppressed within ``BEARING_CAUTIOUS_DIST``), **stuck escape**
(reverse / rotate + replan; skip goal after repeated escapes), **skip unreachable** goals
when A* stays absent (threshold ``MAX_BEARING_STEPS``), **wander** wall-follow after repeated
skips with no path until A* recovers, per-step **nav CSV** next to each MP4, and periodic summaries.
Spawn is a random valid pose with A* connectivity to at least one room goal.

Requires GPU Isaac Sim / OmniGibson (same stack as render_robot_views.py).
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch as th

_REPO_ROOT = Path(__file__).resolve().parent
_OG_SRC = _REPO_ROOT / "BEHAVIOR-1K" / "OmniGibson"
if _OG_SRC.is_dir() and str(_OG_SRC) not in sys.path:
    sys.path.insert(0, str(_OG_SRC))

ROBOT_NAME = "fpv_turtlebot"
ACTION_FREQ = 30
DEFAULT_MAX_SECONDS = 60.0

# Navigation (match autonomous_nav_60s.py)
LOOK_AHEAD_M = 0.75
REPLAN_EVERY_STEPS = 30
PATH_DEVIATION_REPLAN_M = 1.2
GOAL_REACH_M = 0.55
KP_ANG = 2.2
V_MAX_NORM = 0.85
W_MAX_NORM = 0.9
LIDAR_MIN_RANGE = 0.05
LIDAR_MAX_RANGE = 10.0
FRONT_HALF_DEG = 35.0
SIDE_DEG = 55.0
SLOW_DIST = 0.40
STOP_DIST = 0.15
POST_LAND_SIM_STEPS = 15
POST_LAND_RENDER_PASSES = 3
DOORWAY_RADIUS_CELLS = 12

# Stuck escape + unreachable bearing (diagnostics / anti wall-oscillation)
STUCK_WINDOW_STEPS = 90
STUCK_DIST_THRESHOLD_M = 0.15
MAX_STUCK_ESCAPES = 5
ESCAPE_REVERSE_STEPS = 25
ESCAPE_REVERSE_V = -0.5
ESCAPE_ROT_STEPS = 25
ESCAPE_ROT_W = 0.9
# > STUCK_WINDOW_STEPS so reverse/rotate escape can fire before skip-to-next-goal
MAX_BEARING_STEPS = 120
SUMMARY_EVERY_STEPS = 150
BEARING_CAUTIOUS_DIST = 0.8
BEARING_MIN_TURN_W = 0.3
WANDER_AFTER_SKIPS = 2
WANDER_CLEAR_FRONT_M = 0.9
WANDER_SIDE_MIN_M = 0.4
WANDER_V = 0.4
WANDER_W = 0.5

# Fall-through detection (match autonomous_nav_60s.py margin)
FALL_THRESHOLD_M = 1.0
MAX_FALL_RECOVERIES = 3
POST_SPAWN_LAND_RETRIES = 3
SPAWN_RANDOM_MAX_TRIES = 60

# Legacy helpers (preset polylines / teleport tooling)
TELEPORT_STEP_M = 0.02
CHAIN_TOUR_MAX_SEGMENTS = 24

# Rs_int hardcoded doorway rectangles (row0, row1, col0, col1) — same as autonomous_nav_60s.py
RS_INT_DOORWAYS: list[tuple[int, int, int, int]] = [
    (54, 59, 28, 38),
    (54, 59, 48, 58),
    (22, 50, 18, 24),
]

# Curated object subset (same as autonomous_nav_60s.py): avoids clutter that blocks paths and
# mismatches trav_map_with_objects vs LiDAR.
NAV_LOAD_OBJECT_CATEGORIES: list[str] = [
    "floors",
    "ceilings",
    "walls",
    "bottom_cabinet",
    "top_cabinet",
    "breakfast_table",
    "coffee_table",
    "chair",
    "sofa",
    "bed",
    "shelf",
    "fridge",
    "stove",
    "countertop",
    "toilet",
    "bathtub",
]

DEFAULT_SCENE_NAMES = [
    "Beechwood_0_int",
    "Rs_int",
    "Benevolence_1_int",
]

DEFAULT_NAV_ROOT = _REPO_ROOT / "output" / "scene_graphs"
DEFAULT_INTERIOR_MARGIN_M = 0.9
LAND_NEAR_RADIUS_M = 2.5


def _prepare_runtime_env(*, headless: bool, data_path: Path | None) -> None:
    if data_path is not None:
        resolved = data_path.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ["OMNIGIBSON_DATA_PATH"] = str(resolved)
    if headless:
        os.environ["OMNIGIBSON_HEADLESS"] = "1"
    if not os.environ.get("OMNI_KIT_ACCEPT_EULA"):
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def choose_floor_index(scene_name: str, n_floors: int) -> int | None:
    lower = scene_name.lower()
    if n_floors <= 1:
        return 0
    if "house_double_floor_upper" in lower or (
        "upper" in lower and "house_double_floor" in lower
    ):
        return min(1, n_floors - 1)
    if "house_double_floor_lower" in lower or (
        "lower" in lower and "house_double_floor" in lower
    ):
        return 0
    return None


def build_env_config(scene_model: str, resolution: int, structure_only: bool) -> dict:
    scene_cfg: dict = {
        "type": "InteractiveTraversableScene",
        "scene_model": scene_model,
        "trav_map_resolution": 0.1,
        "default_erosion_radius": 0.18,
        "trav_map_with_objects": True,
        "num_waypoints": 1,
        "waypoint_resolution": 0.2,
        "scene_source": "OG",
        "include_robots": True,
    }
    if structure_only:
        from omnigibson.utils.constants import STRUCTURE_CATEGORIES

        scene_cfg["load_object_categories"] = list(STRUCTURE_CATEGORIES)
    else:
        scene_cfg["load_object_categories"] = list(NAV_LOAD_OBJECT_CATEGORIES)

    return {
        "env": {
            "device": None,
            "automatic_reset": False,
            "flatten_action_space": False,
            "flatten_obs_space": False,
            "use_external_obs": False,
            "initial_pos_z_offset": 0.5,
            "external_sensors": None,
        },
        "render": {
            "viewer_width": 640,
            "viewer_height": 480,
        },
        "scene": scene_cfg,
        "robots": [
            {
                "type": "Turtlebot",
                "name": ROBOT_NAME,
                "obs_modalities": ["scan", "rgb"],
                "include_sensor_names": None,
                "exclude_sensor_names": None,
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": resolution,
                            "image_width": resolution,
                        }
                    },
                    "ScanSensor": {
                        "sensor_kwargs": {
                            "min_range": LIDAR_MIN_RANGE,
                            "max_range": LIDAR_MAX_RANGE,
                        }
                    },
                },
                "scale": 1.0,
                "self_collision": False,
                "action_normalize": True,
                "action_type": "continuous",
                "controller_config": {"base": {"name": "DifferentialDriveController"}},
            }
        ],
        "objects": [],
        "task": {"type": "DummyTask"},
        "wrapper": {"type": None},
    }


def extract_modality_from_obs(robot_obs: dict, key: str):
    for _sensor_name, sensor_obs in robot_obs.items():
        if isinstance(sensor_obs, dict) and key in sensor_obs:
            return sensor_obs[key]
    return None


def extract_scan_from_obs(robot_obs: dict) -> th.Tensor | None:
    for _sensor_name, sensor_obs in robot_obs.items():
        if isinstance(sensor_obs, dict) and "scan" in sensor_obs:
            s = sensor_obs["scan"]
            if isinstance(s, th.Tensor):
                return s.squeeze().flatten().float()
            return th.as_tensor(s, dtype=th.float32).squeeze().flatten()
    return None


def tensor_to_numpy(x):
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rgb_to_bgr_uint8(rgb_tensor) -> np.ndarray:
    arr = tensor_to_numpy(rgb_tensor)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected HxWx>=3 RGB, got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# --- Navigation geometry ---

def wrap_angle_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def robot_xy_yaw(robot) -> tuple[float, float, float]:
    import omnigibson.utils.transform_utils as T

    pos, quat = robot.get_position_orientation()
    x = float(pos[0].item())
    y = float(pos[1].item())
    yaw = float(T.z_angle_from_quat(quat).item())
    return x, y, yaw


def distance_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def compute_global_path(
    env, robot, floor: int, start_xy: tuple[float, float], goal_xy: tuple[float, float]
):
    sx, sy = start_xy
    gx, gy = goal_xy
    path, geo = env.scene.get_shortest_path(
        floor,
        th.tensor([sx, sy]),
        th.tensor([gx, gy]),
        entire_path=True,
        robot=robot,
    )
    if path is None:
        return None, None
    return path, geo


def trav_floor_z(env, floor: int = 0) -> float:
    h = env.scene.trav_map.floor_heights[floor]
    return float(h.item()) if isinstance(h, th.Tensor) else float(h)


def robot_height_z(robot) -> float:
    pos, _ = robot.get_position_orientation()
    return float(pos[2].item())


def robot_below_floor(robot, floor_z: float, threshold_m: float = FALL_THRESHOLD_M) -> bool:
    return robot_height_z(robot) < floor_z - threshold_m


def closest_path_index(path: th.Tensor, x: float, y: float) -> int:
    if path is None or path.numel() == 0:
        return 0
    pts = path[:, :2].float()
    dx = pts[:, 0] - x
    dy = pts[:, 1] - y
    d2 = dx * dx + dy * dy
    return int(th.argmin(d2).item())


def path_deviation_m(path: th.Tensor, x: float, y: float) -> float:
    if path is None or path.numel() == 0:
        return 0.0
    i = closest_path_index(path, x, y)
    px = float(path[i, 0].item())
    py = float(path[i, 1].item())
    return distance_xy(x, y, px, py)


def lookahead_point(path: th.Tensor, x: float, y: float, look_ahead: float) -> tuple[float, float] | None:
    if path is None or path.numel() == 0:
        return None
    n = path.shape[0]
    i = closest_path_index(path, x, y)
    px, py = float(path[i, 0].item()), float(path[i, 1].item())
    remaining = look_ahead
    j = i
    while j < n - 1:
        nx = float(path[j + 1, 0].item())
        ny = float(path[j + 1, 1].item())
        seg = distance_xy(px, py, nx, ny)
        if seg >= remaining:
            t = remaining / seg if seg > 1e-6 else 0.0
            return px + t * (nx - px), py + t * (ny - py)
        remaining -= seg
        px, py = nx, ny
        j += 1
    return px, py


def pure_pursuit_cmd(
    x: float,
    y: float,
    yaw: float,
    path: th.Tensor,
    *,
    look_ahead_m: float | None = None,
    v_max_norm: float | None = None,
) -> tuple[float, float]:
    la = LOOK_AHEAD_M if look_ahead_m is None else look_ahead_m
    vmax = V_MAX_NORM if v_max_norm is None else v_max_norm
    target = lookahead_point(path, x, y, la)
    if target is None:
        return 0.0, 0.0
    tx, ty = target
    heading = math.atan2(ty - y, tx - x)
    err = wrap_angle_pi(heading - yaw)
    w = max(-1.0, min(1.0, KP_ANG * err / math.pi))
    align = max(0.0, math.cos(err))
    v = vmax * align
    if abs(err) > math.radians(75):
        v *= 0.25
    return max(-1.0, min(1.0, v)), max(-1.0, min(1.0, w))


def bearing_to_goal_cmd(x: float, y: float, yaw: float, gx: float, gy: float) -> tuple[float, float]:
    """Turn-on-the-spot until aligned, then drive forward (matches autonomous_nav_60s.py)."""
    heading = math.atan2(gy - y, gx - x)
    err = wrap_angle_pi(heading - yaw)
    w = max(-1.0, min(1.0, KP_ANG * err / math.pi))
    if abs(err) > math.radians(22):
        return 0.0, w
    return V_MAX_NORM, 0.0


def normalized_scan_to_meters(scan_norm: th.Tensor) -> th.Tensor:
    return scan_norm * (LIDAR_MAX_RANGE - LIDAR_MIN_RANGE) + LIDAR_MIN_RANGE


def lidar_sector_mins(dist_m: th.Tensor) -> tuple[float, float, float]:
    """Min LiDAR range (m) in front / left / right sectors (same geometry as ``lidar_avoidance_cmd``)."""
    n = dist_m.shape[0]
    if n < 8:
        return LIDAR_MAX_RANGE, LIDAR_MAX_RANGE, LIDAR_MAX_RANGE
    angles = th.linspace(-math.pi, math.pi, n, device=dist_m.device, dtype=dist_m.dtype)
    deg = angles * (180.0 / math.pi)
    front = (deg.abs() <= FRONT_HALF_DEG) | (deg.abs() >= 360.0 - FRONT_HALF_DEG)
    left = (deg > FRONT_HALF_DEG) & (deg < SIDE_DEG)
    right = (deg < -FRONT_HALF_DEG) & (deg > -SIDE_DEG)
    d_front = float(th.min(dist_m[front]).item()) if bool(front.any()) else LIDAR_MAX_RANGE
    d_left = float(th.min(dist_m[left]).item()) if bool(left.any()) else LIDAR_MAX_RANGE
    d_right = float(th.min(dist_m[right]).item()) if bool(right.any()) else LIDAR_MAX_RANGE
    return d_front, d_left, d_right


def lidar_avoidance_cmd(dist_m: th.Tensor) -> tuple[float, float] | None:
    n = dist_m.shape[0]
    if n < 8:
        return None
    d_front, d_left, d_right = lidar_sector_mins(dist_m)
    if d_front > SLOW_DIST:
        return None
    if d_front < STOP_DIST:
        turn = 1.0 if d_left > d_right else -1.0
        return 0.0, turn * W_MAX_NORM
    gap = (d_left - d_right) / max(LIDAR_MAX_RANGE, 1e-6)
    w_avoid = max(-1.0, min(1.0, gap * 2.5))
    v_scale = max(0.15, (d_front - STOP_DIST) / max(SLOW_DIST - STOP_DIST, 1e-6))
    return v_scale, w_avoid


def blend_avoidance(v_pp: float, w_pp: float, avoid: tuple[float, float] | None) -> tuple[float, float]:
    if avoid is None:
        return max(-1.0, min(1.0, v_pp)), max(-1.0, min(1.0, w_pp))
    v_sc, w_av = avoid
    if v_sc == 0.0 and abs(w_av) > 0.5:
        return 0.0, max(-1.0, min(1.0, w_av))
    v = max(-1.0, min(1.0, v_pp * v_sc))
    w = max(-1.0, min(1.0, w_pp + w_av))
    return v, w


def is_stuck_from_hist(hist: deque[tuple[float, float]]) -> bool:
    if len(hist) < STUCK_WINDOW_STEPS:
        return False
    xs = [p[0] for p in hist]
    ys = [p[1] for p in hist]
    return (max(xs) - min(xs) < STUCK_DIST_THRESHOLD_M) and (
        max(ys) - min(ys) < STUCK_DIST_THRESHOLD_M
    )


# --- Doors / trav map ---

def open_all_doors(env) -> None:
    import omnigibson as og
    from omnigibson.object_states import Open

    doors = list(env.scene.object_registry("category", "door", set()))
    sliding = list(env.scene.object_registry("category", "sliding_door", set()))
    all_doors = doors + sliding
    opened = 0
    for door in all_doors:
        if Open in door.states:
            ok = door.states[Open].set_value(True, fully=True)
            if ok:
                opened += 1
        try:
            door.visual_only = True
        except Exception:
            pass
    og.log.info(f"Opened / visual_only {opened}/{len(all_doors)} doors.")


def paint_doorways_from_doors(env, floor: int = 0) -> int:
    import omnigibson as og

    trav = env.scene.trav_map
    fmap = trav.floor_map[floor]
    map_sz = fmap.shape[0]
    doors = list(env.scene.object_registry("category", "door", set()))
    sliding = list(env.scene.object_registry("category", "sliding_door", set()))
    count = 0
    for door in doors + sliding:
        pos, _ = door.get_position_orientation()
        xy = th.tensor([float(pos[0]), float(pos[1])])
        rc = trav.world_to_map(xy)
        r, c = int(rc[0].item()), int(rc[1].item())
        r0 = max(0, r - DOORWAY_RADIUS_CELLS)
        r1 = min(map_sz - 1, r + DOORWAY_RADIUS_CELLS)
        c0 = max(0, c - DOORWAY_RADIUS_CELLS)
        c1 = min(map_sz - 1, c + DOORWAY_RADIUS_CELLS)
        fmap[r0 : r1 + 1, c0 : c1 + 1] = 255
        count += 1
    og.log.info(f"Painted {count} doorway(s) from door objects.")
    return count


def paint_rs_int_hardcoded_doorways(env, floor: int = 0) -> None:
    import omnigibson as og

    trav = env.scene.trav_map.floor_map[floor]
    for r0, r1, c0, c1 in RS_INT_DOORWAYS:
        trav[r0 : r1 + 1, c0 : c1 + 1] = 255
    og.log.info(f"Painted {len(RS_INT_DOORWAYS)} Rs_int hardcoded doorway rectangle(s).")


def prepare_trav_map_doorways(env, scene_model: str, floor: int = 0) -> None:
    import omnigibson as og

    n = paint_doorways_from_doors(env, floor=floor)
    if n == 0:
        og.log.warning(
            "No door objects found in registry; trav map uses hardcoded DOORWAYS only."
        )
    if scene_model == "Rs_int":
        paint_rs_int_hardcoded_doorways(env, floor=floor)


def stabilize_after_land(env, robot_name: str) -> None:
    import omnigibson as og

    zero = th.tensor([0.0, 0.0], dtype=th.float32)
    for _ in range(POST_LAND_SIM_STEPS):
        env.step({robot_name: zero})
    for _ in range(POST_LAND_RENDER_PASSES):
        og.sim.render()


# --- Land near target xy (room centroid) ---

def _erode_trav_map_pixels(trav_map_uint8: np.ndarray, map_resolution: float, radius_m: float) -> np.ndarray:
    radius_px = max(1, int(math.ceil(radius_m / max(map_resolution, 1e-6))))
    kernel = np.ones((radius_px, radius_px), dtype=np.uint8)
    return cv2.erode(trav_map_uint8.astype(np.uint8), kernel)


def yaw_toward_nearest_content_object(env, robot, rx: float, ry: float) -> float | None:
    from omnigibson.robots.robot_base import m as robot_macros
    from omnigibson.utils.constants import STRUCTURAL_DOOR_CATEGORIES, STRUCTURE_CATEGORIES

    best_d2 = float("inf")
    best: tuple[float, float] | None = None
    min_dist2 = 0.15 * 0.15
    for obj in env.scene.object_registry.objects:
        if obj is robot:
            continue
        cat = getattr(obj, "category", None) or ""
        if cat in STRUCTURE_CATEGORIES or cat in STRUCTURAL_DOOR_CATEGORIES:
            continue
        if cat == robot_macros.ROBOT_CATEGORY:
            continue
        try:
            pos, _ = obj.get_position_orientation()
        except Exception:
            continue
        ox = float(pos[0].item()) if isinstance(pos[0], th.Tensor) else float(pos[0])
        oy = float(pos[1].item()) if isinstance(pos[1], th.Tensor) else float(pos[1])
        d2 = (ox - rx) ** 2 + (oy - ry) ** 2
        if d2 < min_dist2:
            continue
        if d2 < best_d2:
            best_d2 = d2
            best = (ox, oy)
    if best is None:
        return None
    return math.atan2(best[1] - ry, best[0] - rx)


def try_land_near_xy(
    env,
    robot,
    floor: int,
    target_xy: tuple[float, float],
    interior_margin_m: float,
    max_radius_m: float,
    max_tries: int,
) -> bool:
    import omnigibson.utils.transform_utils as T
    from omnigibson.utils.sim_utils import land_object, test_valid_pose

    z_off = env.initial_pos_z_offset
    trav = env.scene.trav_map
    tx, ty = target_xy
    base = th.clone(trav.floor_map[floor])
    base_np = base.cpu().numpy().astype(np.uint8)
    inner_np = _erode_trav_map_pixels(base_np, float(trav.map_resolution), interior_margin_m)
    inner = th.as_tensor(inner_np, dtype=base.dtype, device=base.device)
    robot_ok = trav._erode_trav_map(th.clone(base), robot=robot)
    combined = inner.clone()
    combined[robot_ok != 255] = 0
    if not bool((combined == 255).any().item()):
        combined = robot_ok

    free_r, free_c = th.where(combined == 255)
    if free_r.numel() == 0:
        return False

    r2_max = max_radius_m * max_radius_m
    cand_idx: list[int] = []
    for i in range(free_r.numel()):
        xy_map = th.tensor([float(free_r[i].item()), float(free_c[i].item())], dtype=th.float32)
        wxy = trav.map_to_world(xy_map)
        rx = float(wxy[0].item())
        ry = float(wxy[1].item())
        if (rx - tx) ** 2 + (ry - ty) ** 2 <= r2_max:
            cand_idx.append(i)

    if not cand_idx:
        for _ in range(max_tries):
            j = int(th.randint(0, free_r.numel(), (1,)).item())
            cand_idx.append(j)

    fh = trav.floor_heights[floor]
    rz0 = float(fh.item()) if isinstance(fh, th.Tensor) else float(fh)

    for _ in range(max_tries):
        j = int(random.choice(cand_idx))
        r_i, c_i = int(free_r[j].item()), int(free_c[j].item())
        xy_map = th.tensor([float(r_i), float(c_i)], dtype=th.float32)
        wxy = trav.map_to_world(xy_map)
        rx = float(wxy[0].item())
        ry = float(wxy[1].item())
        rpos = th.tensor([rx, ry, rz0], dtype=th.float32)
        yaw = yaw_toward_nearest_content_object(env, robot, rx, ry)
        if yaw is None:
            yaw = random.uniform(0, 2 * math.pi)
        quat = T.euler2quat(th.tensor([0.0, 0.0, yaw]))
        if test_valid_pose(robot, rpos, quat, z_off):
            land_object(robot, rpos, quat, z_off)
            return True
    return False


# --- Trajectory planning from nav_paths.json ---

def load_nav_envelope(nav_path: Path) -> dict[str, Any] | None:
    if not nav_path.is_file():
        return None
    with open(nav_path, encoding="utf-8") as f:
        return json.load(f)


def build_room_graph_and_direct_paths(
    envelope: dict[str, Any], floor_id: int
) -> tuple[dict[str, list[tuple[str, float]]], dict[tuple[str, str], list[list[float]]]]:
    """Undirected weighted adjacency + directed waypoint cache (forward keys only)."""
    neighbors: dict[str, list[tuple[str, float]]] = {}
    direct_wps: dict[tuple[str, str], list[list[float]]] = {}

    for p in envelope.get("paths", []):
        if p.get("status") != "ok":
            continue
        if int(p.get("floor_id", 0)) != floor_id:
            continue
        ra = p["start"]["room"]
        rb = p["end"]["room"]
        wps = p.get("waypoints") or []
        if len(wps) < 2:
            continue
        L = float(p.get("path_length_m") or 0.0)
        tup = (ra, rb)
        direct_wps[tup] = [list(map(float, xy)) for xy in wps]
        neighbors.setdefault(ra, []).append((rb, L))
        neighbors.setdefault(rb, []).append((ra, L))

    return neighbors, direct_wps


def get_waypoints_between(
    a: str, b: str, direct_wps: dict[tuple[str, str], list[list[float]]]
) -> list[list[float]] | None:
    if (a, b) in direct_wps:
        return [list(xy) for xy in direct_wps[(a, b)]]
    if (b, a) in direct_wps:
        rev = direct_wps[(b, a)]
        return [list(rev[i]) for i in range(len(rev) - 1, -1, -1)]
    return None


def dijkstra_distances(
    start: str, neighbors: dict[str, list[tuple[str, float]]]
) -> dict[str, float]:
    dist: dict[str, float] = {start: 0.0}
    pq: list[tuple[float, str]] = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")) + 1e-9:
            continue
        for v, w in neighbors.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def pick_farthest_unvisited_room(
    current: str,
    neighbors: dict[str, list[tuple[str, float]]],
    all_rooms: set[str],
    visited: set[str],
) -> str | None:
    dist = dijkstra_distances(current, neighbors)
    best_room: str | None = None
    best_d = -1.0
    candidates = [r for r in all_rooms if r != current and r not in visited and r in dist]
    if not candidates:
        visited.clear()
        candidates = [r for r in all_rooms if r != current and r in dist]
    for r in candidates:
        d = dist[r]
        if d > best_d:
            best_d = d
            best_room = r
        elif abs(d - best_d) < 1e-6 and best_room is not None:
            if random.random() < 0.5:
                best_room = r
    return best_room


def rooms_on_floor_from_nav(envelope: dict[str, Any], floor_id: int) -> set[str]:
    rooms: set[str] = set()
    for p in envelope.get("paths", []):
        if p.get("status") != "ok":
            continue
        if int(p.get("floor_id", 0)) != floor_id:
            continue
        rooms.add(p["start"]["room"])
        rooms.add(p["end"]["room"])
    return rooms


def room_centroids_from_nav(
    envelope: dict[str, Any], floor_id: int
) -> dict[str, tuple[float, float]]:
    """Room id -> world (x, y) from nav_paths.json path endpoints (last wins if duplicate)."""
    out: dict[str, tuple[float, float]] = {}
    for p in envelope.get("paths", []):
        if p.get("status") != "ok":
            continue
        if int(p.get("floor_id", 0)) != floor_id:
            continue
        for key in ("start", "end"):
            node = p.get(key) or {}
            rid = node.get("room")
            w = node.get("world_xy")
            if rid and w and len(w) >= 2:
                out[str(rid)] = (float(w[0]), float(w[1]))
    return out


def snap_to_trav(
    env,
    floor: int,
    xy: tuple[float, float],
    max_radius_cells: int = 20,
) -> tuple[float, float]:
    """Snap a world (x,y) to the nearest traversable trav-map cell (value 255)."""
    trav = env.scene.trav_map
    fmap = trav.floor_map[floor]
    sx, sy = float(xy[0]), float(xy[1])
    rc = trav.world_to_map(th.tensor([sx, sy], dtype=th.float32))
    r0, c0 = int(rc[0].item()), int(rc[1].item())
    h, w = int(fmap.shape[0]), int(fmap.shape[1])

    def cell_val(rr: int, cc: int) -> int | None:
        if 0 <= rr < h and 0 <= cc < w:
            return int(fmap[rr, cc].item())
        return None

    if cell_val(r0, c0) == 255:
        return (sx, sy)

    q: deque[tuple[int, int, int]] = deque([(r0, c0, 0)])
    visited = {(r0, c0)}
    dirs8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while q:
        r, c, d = q.popleft()
        if d > max_radius_cells:
            continue
        if cell_val(r, c) == 255:
            xy_map = th.tensor([float(r), float(c)], dtype=th.float32)
            wxy = trav.map_to_world(xy_map)
            return (float(wxy[0].item()), float(wxy[1].item()))
        for dr, dc in dirs8:
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            visited.add((nr, nc))
            if cell_val(nr, nc) is not None:
                q.append((nr, nc, d + 1))
    return (sx, sy)


def nearest_room_to_xy(
    x: float,
    y: float,
    room_xy: dict[str, tuple[float, float]],
    candidates: set[str] | None = None,
) -> str | None:
    keys = candidates if candidates is not None else set(room_xy.keys())
    best: str | None = None
    best_d = float("inf")
    for rid in keys:
        if rid not in room_xy:
            continue
        cx, cy = room_xy[rid]
        d = distance_xy(x, y, cx, cy)
        if d < best_d:
            best_d = d
            best = rid
    return best


def _path_polyline_length_m(path: th.Tensor | None) -> str:
    if path is None:
        return "N/A"
    n = int(path.shape[0])
    if n < 2:
        return "0.00"
    total = 0.0
    for i in range(n - 1):
        dx = float(path[i + 1, 0].item() - path[i, 0].item())
        dy = float(path[i + 1, 1].item() - path[i, 1].item())
        total += math.hypot(dx, dy)
    return f"{total:.2f}"


def pick_next_goal_with_path(
    env,
    robot,
    floor: int,
    xy: tuple[float, float],
    current_room: str,
    neighbors: dict[str, list[tuple[str, float]]],
    all_rooms: set[str],
    visited: set[str],
    room_xy: dict[str, tuple[float, float]],
) -> tuple[str, float, float, th.Tensor | None]:
    """Pick farthest unvisited room (graph distance) that has an A* path from ``xy``; else best effort."""
    dist = dijkstra_distances(current_room, neighbors)
    candidates = [
        r
        for r in all_rooms
        if r != current_room and r not in visited and r in dist and r in room_xy
    ]
    if not candidates:
        visited.clear()
        candidates = [r for r in all_rooms if r != current_room and r in dist and r in room_xy]
    candidates.sort(key=lambda r: dist.get(r, 0.0), reverse=True)
    x0, y0 = xy
    trav = env.scene.trav_map
    for r in candidates:
        gx, gy = room_xy[r]
        path, _ = compute_global_path(env, robot, floor, xy, (gx, gy))
        d_straight = distance_xy(x0, y0, gx, gy)
        path_ok = "ok" if path is not None else "None"
        path_len = _path_polyline_length_m(path)
        print(
            f"  [goal_check] room={r} goal=({gx:.2f},{gy:.2f}) "
            f"d_straight={d_straight:.2f}m path={path_ok} path_len={path_len}m"
        )
        if path is not None:
            rc = trav.world_to_map(th.tensor([gx, gy], dtype=th.float32))
            cell_val = int(
                trav.floor_map[floor][int(rc[0].item()), int(rc[1].item())].item()
            )
            print(f"  [goal_diag] selected {r} cell_val={cell_val}")
            return r, gx, gy, path
    if not candidates:
        gx, gy = room_xy.get(current_room, (xy[0], xy[1]))
        path, _ = compute_global_path(env, robot, floor, xy, (gx, gy))
        rc = trav.world_to_map(th.tensor([gx, gy], dtype=th.float32))
        cell_val = int(trav.floor_map[floor][int(rc[0].item()), int(rc[1].item())].item())
        print(f"  [goal_diag] selected {current_room} cell_val={cell_val}")
        return current_room, gx, gy, path
    r = candidates[0]
    gx, gy = room_xy[r]
    path, _ = compute_global_path(env, robot, floor, xy, (gx, gy))
    rc = trav.world_to_map(th.tensor([gx, gy], dtype=th.float32))
    cell_val = int(trav.floor_map[floor][int(rc[0].item()), int(rc[1].item())].item())
    print(f"  [goal_diag] selected {r} cell_val={cell_val}")
    return r, gx, gy, path


def spawn_random_connected_to_centroids(
    env,
    robot,
    floor: int,
    goal_xys: list[tuple[float, float]],
    max_tries: int = SPAWN_RANDOM_MAX_TRIES,
) -> bool:
    """Random valid pose that has an A* path to at least one goal (same idea as spawn_near_goals)."""
    import omnigibson.utils.transform_utils as T
    from omnigibson.utils.sim_utils import land_object, test_valid_pose

    z_off = env.initial_pos_z_offset
    for _ in range(max_tries):
        _, rpos = env.scene.get_random_point(floor=floor, robot=robot)
        yaw = random.uniform(0.0, 2.0 * math.pi)
        rquat = T.euler2quat(th.tensor([0.0, 0.0, yaw]))
        if not test_valid_pose(robot, rpos, rquat, z_off):
            continue
        rx, ry = float(rpos[0].item()), float(rpos[1].item())
        for gx, gy in goal_xys:
            path, _ = compute_global_path(env, robot, floor, (rx, ry), (gx, gy))
            if path is not None:
                land_object(robot, rpos, rquat, z_off)
                return True
    return False


def resolve_floor(scene_name: str, n_floors: int, floor_arg: int | None) -> int:
    if floor_arg is not None:
        return int(floor_arg)
    mode = choose_floor_index(scene_name, n_floors)
    if mode is not None:
        return int(mode)
    return 0


def collect_coarse_tour_polyline(
    envelope: dict[str, Any],
    floor: int,
    neighbors: dict[str, list[tuple[str, float]]],
    all_rooms: set[str],
    direct_wps: dict[tuple[str, str], list[list[float]]],
    start_room: str,
    *,
    max_segments: int = CHAIN_TOUR_MAX_SEGMENTS,
) -> list[list[float]]:
    """
    Same room tour as the video rollout: chain segment waypoints from nav_paths.json into one coarse
    polyline (world xy), without densification.
    """
    _ = envelope, floor
    visited: set[str] = set()
    current_room = start_room
    next_room = pick_farthest_unvisited_room(current_room, neighbors, all_rooms, visited)
    if next_room is None:
        return []

    coarse: list[list[float]] = []
    segments = 0

    while next_room is not None and segments < max_segments:
        segment_wps = get_waypoints_between(current_room, next_room, direct_wps)
        if not segment_wps:
            next_room = pick_farthest_unvisited_room(
                current_room, neighbors, all_rooms, {current_room} | visited
            )
            if next_room is None:
                break
            segment_wps = get_waypoints_between(current_room, next_room, direct_wps)
        if not segment_wps:
            break

        for pt in segment_wps:
            xy = [float(pt[0]), float(pt[1])]
            if coarse and distance_xy(xy[0], xy[1], coarse[-1][0], coarse[-1][1]) < 1e-5:
                continue
            coarse.append(xy)

        visited.add(next_room)
        current_room = next_room
        next_room = pick_farthest_unvisited_room(current_room, neighbors, all_rooms, visited)
        segments += 1

    if len(coarse) < 2:
        return []
    return coarse


def interpolate_waypoints(
    waypoints: list[list[float]], step_m: float = TELEPORT_STEP_M
) -> list[tuple[float, float, float]]:
    """Densify coarse [x, y] polyline to ~``step_m`` spacing; yaw faces the next point."""
    if len(waypoints) < 2:
        if len(waypoints) == 1:
            return [(float(waypoints[0][0]), float(waypoints[0][1]), 0.0)]
        return []
    out: list[tuple[float, float, float]] = []
    min_d = max(step_m * 0.08, 1e-4)

    def append_unique(x: float, y: float, yaw: float) -> None:
        if out and distance_xy(x, y, out[-1][0], out[-1][1]) < min_d:
            out[-1] = (x, y, yaw)
            return
        out.append((x, y, yaw))

    for i in range(len(waypoints) - 1):
        x0, y0 = float(waypoints[i][0]), float(waypoints[i][1])
        x1, y1 = float(waypoints[i + 1][0]), float(waypoints[i + 1][1])
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            continue
        yaw = math.atan2(dy, dx)
        dist_along = 0.0
        while dist_along < seg_len - 1e-9:
            t = dist_along / seg_len
            append_unique(x0 + t * dx, y0 + t * dy, yaw)
            dist_along += step_m
        append_unique(x1, y1, yaw)

    return out


def chain_tour_waypoints(
    envelope: dict[str, Any],
    floor: int,
    neighbors: dict[str, list[tuple[str, float]]],
    all_rooms: set[str],
    direct_wps: dict[tuple[str, str], list[list[float]]],
    start_room: str,
    *,
    step_m: float = TELEPORT_STEP_M,
    max_segments: int = CHAIN_TOUR_MAX_SEGMENTS,
) -> list[tuple[float, float, float]]:
    """Chain tour segments then densify (legacy helper for tooling / experiments)."""
    coarse = collect_coarse_tour_polyline(
        envelope, floor, neighbors, all_rooms, direct_wps, start_room, max_segments=max_segments
    )
    if not coarse:
        return []
    return interpolate_waypoints(coarse, step_m)


# --- Video + main loop per scene ---


def _resolve_ffmpeg_tool(name: str) -> str | None:
    """``shutil.which`` plus common install paths (conda/headless often omits Homebrew from PATH)."""
    found = shutil.which(name)
    if found:
        return found
    for d in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        p = Path(d) / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def make_video_writer(path: Path, fps: float, width: int, height: int) -> tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if w.isOpened():
        return w, path
    path_avi = path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w = cv2.VideoWriter(str(path_avi), fourcc, float(fps), (width, height))
    if not w.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path} or {path_avi}")
    return w, path_avi


def finalize_mp4_for_apple_players(path: Path) -> None:
    """OpenCV writes mp4v (MPEG-4 Part 2); re-encode to H.264 Baseline for QuickTime / Finder / Safari."""
    if path.suffix.lower() != ".mp4" or not path.is_file():
        return
    ffmpeg_bin = _resolve_ffmpeg_tool("ffmpeg")
    if ffmpeg_bin is None:
        print(
            f"  [WARN] ffmpeg not found; {path.name} may not play in QuickTime or IDE preview. "
            "Install ffmpeg (e.g. brew install ffmpeg) and re-run, or: ./transcode_robot_videos_h264.sh"
        )
        return
    tmp = path.with_suffix(".apple_h264.tmp.mp4")
    # Upscale tiny FPV (e.g. 256^2) so QuickTime / IDE previews accept the file.
    vf: list[str] = []
    ffprobe_bin = _resolve_ffmpeg_tool("ffprobe")
    if ffprobe_bin:
        try:
            pr = subprocess.run(
                [
                    ffprobe_bin,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0:s=x",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if pr.returncode == 0 and pr.stdout.strip():
                parts = pr.stdout.strip().split("x")
                if len(parts) == 2:
                    w0, h0 = int(parts[0]), int(parts[1])
                    if min(w0, h0) < 480:
                        if w0 >= h0:
                            vf = ["-vf", "scale=480:-2:flags=neighbor"]
                        else:
                            vf = ["-vf", "scale=-2:480:flags=neighbor"]
        except (ValueError, OSError):
            pass
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=mono",
        "-i",
        str(path),
        "-shortest",
        "-map",
        "1:v:0",
        "-map",
        "0:a:0",
        *vf,
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "48k",
        "-ac",
        "1",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        tmp.replace(path)
        if sys.platform == "darwin":
            subprocess.run(["xattr", "-c", str(path)], check=False, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"  [WARN] ffmpeg finalize failed for {path}: {e}")
        if tmp.is_file():
            tmp.unlink(missing_ok=True)


def run_scene_video(
    scene_model: str,
    nav_root: Path,
    *,
    resolution: int,
    max_steps: int,
    fps: float,
    output_dir: Path,
    structure_only: bool,
    floor_arg: int | None,
    interior_margin_m: float,
    seed: int | None,
) -> bool:
    import omnigibson as og

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

    nav_json = nav_root / scene_model / "nav_paths.json"

    envelope = load_nav_envelope(nav_json)
    if envelope is None:
        print(f"  [{scene_model}] missing {nav_json}, skip")
        return False

    env = og.Environment(configs=build_env_config(scene_model, resolution, structure_only))
    writer: cv2.VideoWriter | None = None
    video_path: Path = output_dir / scene_model / f"fpv_{scene_model}.mp4"
    nav_csv_file = None
    step = 0
    ok = False

    try:
        obs, _ = env.reset()
        open_all_doors(env)
        for _ in range(5):
            og.sim.step()

        robot = env.robots[0]
        assert robot.name == ROBOT_NAME
        n_floors = env.scene.trav_map.n_floors
        floor = resolve_floor(scene_model, n_floors, floor_arg)
        floor = max(0, min(floor, n_floors - 1))

        prepare_trav_map_doorways(env, scene_model, floor=floor)
        floor_z = trav_floor_z(env, floor)

        neighbors, _direct_wps = build_room_graph_and_direct_paths(envelope, floor)
        all_rooms = rooms_on_floor_from_nav(envelope, floor)
        if len(all_rooms) < 2:
            print(f"  [{scene_model}] fewer than 2 rooms on floor {floor}, skip")
            return False

        room_xy = room_centroids_from_nav(envelope, floor)
        for rid, (cx, cy) in list(room_xy.items()):
            sx, sy = snap_to_trav(env, floor, (cx, cy))
            if (sx, sy) != (cx, cy):
                delta = distance_xy(cx, cy, sx, sy)
                print(
                    f"  [{scene_model}] snap centroid {rid}: "
                    f"({cx:.2f},{cy:.2f}) -> ({sx:.2f},{sy:.2f}) delta={delta:.2f}m"
                )
            room_xy[rid] = (sx, sy)
        if len(room_xy) < 2:
            print(f"  [{scene_model}] insufficient room centroids in nav_paths, skip")
            return False

        goal_xys = list(room_xy.values())
        if not spawn_random_connected_to_centroids(
            env, robot, floor, goal_xys, max_tries=SPAWN_RANDOM_MAX_TRIES
        ):
            print(
                f"  [{scene_model}] could not spawn at random pose with A* to any room goal, skip"
            )
            return False

        stabilize_after_land(env, ROBOT_NAME)
        obs, _ = env.get_obs()

        spawn_retry = 0
        while robot_below_floor(robot, floor_z) and spawn_retry < POST_SPAWN_LAND_RETRIES:
            print(
                f"  [{scene_model}] [WARN] Robot below floor "
                f"(z={robot_height_z(robot):.2f}, floor_z={floor_z:.2f}); re-spawn "
                f"({spawn_retry + 1}/{POST_SPAWN_LAND_RETRIES})"
            )
            if not spawn_random_connected_to_centroids(
                env, robot, floor, goal_xys, max_tries=SPAWN_RANDOM_MAX_TRIES
            ):
                print(f"  [{scene_model}] re-spawn failed, skip")
                return False
            stabilize_after_land(env, ROBOT_NAME)
            obs, _ = env.get_obs()
            spawn_retry += 1
        if robot_below_floor(robot, floor_z):
            print(
                f"  [{scene_model}] robot still below floor after {POST_SPAWN_LAND_RETRIES} "
                "re-spawn attempt(s), skip"
            )
            return False

        x, y, yaw = robot_xy_yaw(robot)
        current_room = nearest_room_to_xy(x, y, room_xy, all_rooms) or random.choice(
            tuple(all_rooms)
        )
        visited: set[str] = set()
        next_room, goal_x, goal_y, path = pick_next_goal_with_path(
            env,
            robot,
            floor,
            (x, y),
            current_room,
            neighbors,
            all_rooms,
            visited,
            room_xy,
        )
        replan_counter = 0
        fall_recoveries = 0
        print(
            f"  [{scene_model}] live A* tour: near_room={current_room!r} -> goal_room={next_room!r} "
            f"goal=({goal_x:.2f},{goal_y:.2f}) path={'ok' if path is not None else 'None'}"
        )

        nav_csv_path = output_dir / scene_model / f"fpv_{scene_model}_nav.csv"
        nav_csv_path.parent.mkdir(parents=True, exist_ok=True)
        nav_csv_file = open(nav_csv_path, "w", newline="", encoding="utf-8")
        nav_writer = csv.writer(nav_csv_file)
        nav_writer.writerow(
            [
                "step",
                "x",
                "y",
                "yaw",
                "goal_x",
                "goal_y",
                "goal_room",
                "d_goal",
                "d_front",
                "d_left",
                "d_right",
                "path_status",
                "v_pp",
                "w_pp",
                "v_cmd",
                "w_cmd",
                "avoid_v",
                "avoid_w",
                "mode",
            ]
        )

        pos_hist: deque[tuple[float, float]] = deque(maxlen=STUCK_WINDOW_STEPS)
        escape_subphase: str | None = None
        escape_remaining = 0
        rot_sign = 1.0
        stuck_escapes_this_goal = 0
        consecutive_no_path_steps = 0
        wander_mode = False
        consecutive_failed_skips = 0

        def skip_to_next_goal(reason: str) -> None:
            nonlocal next_room, goal_x, goal_y, path, replan_counter, current_room
            nonlocal consecutive_no_path_steps, stuck_escapes_this_goal
            nonlocal escape_subphase, escape_remaining, pos_hist
            nonlocal wander_mode, consecutive_failed_skips
            print(reason)
            visited.add(next_room)
            x2, y2, _ = robot_xy_yaw(robot)
            current_room = nearest_room_to_xy(x2, y2, room_xy, all_rooms) or current_room
            next_room, goal_x, goal_y, path = pick_next_goal_with_path(
                env,
                robot,
                floor,
                (x2, y2),
                current_room,
                neighbors,
                all_rooms,
                visited,
                room_xy,
            )
            replan_counter = 0
            consecutive_no_path_steps = 0
            stuck_escapes_this_goal = 0
            escape_subphase = None
            escape_remaining = 0
            pos_hist.clear()
            if path is None:
                consecutive_failed_skips += 1
                if consecutive_failed_skips >= WANDER_AFTER_SKIPS:
                    wander_mode = True
                    print(
                        f"  [{scene_model}] wander mode on ({consecutive_failed_skips} skips with no A*); "
                        "wall-follow until replan succeeds"
                    )
            else:
                consecutive_failed_skips = 0
                wander_mode = False
            print(
                f"  [{scene_model}] skip -> goal_room={next_room!r} "
                f"goal=({goal_x:.2f},{goal_y:.2f}) path={'ok' if path is not None else 'None'}"
            )

        def write_rgb_frame() -> None:
            nonlocal obs, writer, video_path
            rgb = extract_modality_from_obs(obs[ROBOT_NAME], "rgb")
            if rgb is not None:
                bgr = rgb_to_bgr_uint8(rgb)
                hh, ww = bgr.shape[0], bgr.shape[1]
                if writer is None:
                    writer, video_path = make_video_writer(video_path, fps, ww, hh)
                writer.write(bgr)

        while step < max_steps:
            write_rgb_frame()

            x, y, yaw = robot_xy_yaw(robot)
            scan_norm = extract_scan_from_obs(obs[ROBOT_NAME])
            if scan_norm is not None:
                dist_m = normalized_scan_to_meters(scan_norm)
                d_front, d_left, d_right = lidar_sector_mins(dist_m)
                avoid = lidar_avoidance_cmd(dist_m)
            else:
                d_front = d_left = d_right = LIDAR_MAX_RANGE
                avoid = None

            replan_counter += 1
            dev = (
                path_deviation_m(path, x, y)
                if path is not None
                else PATH_DEVIATION_REPLAN_M + 1.0
            )
            if path is None or replan_counter >= REPLAN_EVERY_STEPS or dev > PATH_DEVIATION_REPLAN_M:
                path, _ = compute_global_path(env, robot, floor, (x, y), (goal_x, goal_y))
                replan_counter = 0
                if path is not None and wander_mode:
                    wander_mode = False
                    consecutive_failed_skips = 0
                    print(f"  [{scene_model}] A* recovered; leaving wander mode")

            if escape_subphase is None and not wander_mode:
                if path is None:
                    consecutive_no_path_steps += 1
                    if consecutive_no_path_steps >= MAX_BEARING_STEPS:
                        skip_to_next_goal(
                            f"  [{scene_model}] no A* path for {MAX_BEARING_STEPS} consecutive steps; "
                            f"skip goal {next_room!r}"
                        )
                else:
                    consecutive_no_path_steps = 0

            path_status = "bearing"
            mode = "bearing"
            v_pp = 0.0
            w_pp = 0.0

            if escape_subphase == "reverse":
                path_status = "escape"
                mode = "reverse"
                v_cmd, w_cmd = ESCAPE_REVERSE_V, 0.0
                v_pp, w_pp = v_cmd, w_cmd
                escape_remaining -= 1
                if escape_remaining <= 0:
                    escape_subphase = "rotate"
                    escape_remaining = ESCAPE_ROT_STEPS
                    rot_sign = 1.0 if d_left >= d_right else -1.0
            elif escape_subphase == "rotate":
                path_status = "escape"
                mode = "rotate"
                v_cmd, w_cmd = 0.0, ESCAPE_ROT_W * rot_sign
                v_pp, w_pp = v_cmd, w_cmd
                escape_remaining -= 1
                if escape_remaining <= 0:
                    escape_subphase = None
                    stuck_escapes_this_goal += 1
                    path, _ = compute_global_path(env, robot, floor, (x, y), (goal_x, goal_y))
                    replan_counter = 0
                    pos_hist.clear()
                    if stuck_escapes_this_goal >= MAX_STUCK_ESCAPES:
                        skip_to_next_goal(
                            f"  [{scene_model}] stuck escape limit ({MAX_STUCK_ESCAPES}) for goal "
                            f"{next_room!r}; advancing"
                        )
            else:
                if wander_mode:
                    path_status = "wander"
                    mode = "wander"
                    if (
                        d_front > WANDER_CLEAR_FRONT_M
                        and d_left > WANDER_SIDE_MIN_M
                        and d_right > WANDER_SIDE_MIN_M
                    ):
                        v_pp, w_pp = WANDER_V, 0.0
                    else:
                        v_pp, w_pp = 0.0, WANDER_W if d_left >= d_right else -WANDER_W
                    v_cmd, w_cmd = blend_avoidance(v_pp, w_pp, avoid)
                elif path is not None:
                    path_status = "astar"
                    mode = "pursuit"
                    v_pp, w_pp = pure_pursuit_cmd(x, y, yaw, path)
                    v_cmd, w_cmd = blend_avoidance(v_pp, w_pp, avoid)
                else:
                    path_status = "bearing"
                    mode = "bearing"
                    if d_front <= BEARING_CAUTIOUS_DIST:
                        _, w_pp = bearing_to_goal_cmd(x, y, yaw, goal_x, goal_y)
                        v_pp = 0.0
                    else:
                        v_pp, w_pp = bearing_to_goal_cmd(x, y, yaw, goal_x, goal_y)
                    v_cmd, w_cmd = blend_avoidance(v_pp, w_pp, avoid)
                    if d_front < SLOW_DIST and abs(w_cmd) < BEARING_MIN_TURN_W:
                        w_cmd = BEARING_MIN_TURN_W if d_left >= d_right else -BEARING_MIN_TURN_W
                        w_cmd = max(-1.0, min(1.0, w_cmd))

                pos_hist.append((x, y))
                if is_stuck_from_hist(pos_hist):
                    trav = env.scene.trav_map
                    rc = trav.world_to_map(th.tensor([goal_x, goal_y], dtype=th.float32))
                    cell_val = int(
                        trav.floor_map[floor][int(rc[0].item()), int(rc[1].item())].item()
                    )
                    print(
                        f"  [{scene_model}] STUCK at ({x:.2f},{y:.2f}) "
                        f"goal=({goal_x:.2f},{goal_y:.2f}) goal_cell={cell_val}"
                    )
                    pos_hist.clear()
                    escape_subphase = "reverse"
                    escape_remaining = ESCAPE_REVERSE_STEPS
                    escape_remaining -= 1
                    path_status = "escape"
                    mode = "reverse"
                    v_cmd, w_cmd = ESCAPE_REVERSE_V, 0.0
                    v_pp, w_pp = v_cmd, w_cmd

            d_goal = distance_xy(x, y, goal_x, goal_y)
            if avoid is not None:
                avoid_v, avoid_w = avoid[0], avoid[1]
            else:
                avoid_v, avoid_w = "", ""
            nav_writer.writerow(
                [
                    step,
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{yaw:.6f}",
                    f"{goal_x:.6f}",
                    f"{goal_y:.6f}",
                    next_room,
                    f"{d_goal:.6f}",
                    f"{d_front:.6f}",
                    f"{d_left:.6f}",
                    f"{d_right:.6f}",
                    path_status,
                    f"{v_pp:.6f}",
                    f"{w_pp:.6f}",
                    f"{v_cmd:.6f}",
                    f"{w_cmd:.6f}",
                    avoid_v,
                    avoid_w,
                    mode,
                ]
            )

            action = {ROBOT_NAME: th.tensor([v_cmd, w_cmd], dtype=th.float32)}
            obs, _, terminated, truncated, _ = env.step(action)
            step += 1

            if step % SUMMARY_EVERY_STEPS == 0:
                t_sim = step / float(fps)
                if wander_mode:
                    path_summ = "wander"
                elif path is not None:
                    path_summ = "astar"
                elif escape_subphase is not None:
                    path_summ = "escape"
                else:
                    path_summ = "none"
                print(
                    f"  [{scene_model}] t={t_sim:.1f}s step={step} pos=({x:.2f},{y:.2f}) "
                    f"goal={next_room}({goal_x:.2f},{goal_y:.2f}) d_goal={d_goal:.1f}m "
                    f"d_front={d_front:.1f}m path={path_summ} v={v_cmd:.2f} w={w_cmd:.2f}"
                )

            if terminated or truncated:
                break

            x, y, yaw = robot_xy_yaw(robot)

            if distance_xy(x, y, goal_x, goal_y) < GOAL_REACH_M:
                visited.add(next_room)
                current_room = next_room
                next_room, goal_x, goal_y, path = pick_next_goal_with_path(
                    env,
                    robot,
                    floor,
                    (x, y),
                    current_room,
                    neighbors,
                    all_rooms,
                    visited,
                    room_xy,
                )
                replan_counter = 0
                consecutive_no_path_steps = 0
                stuck_escapes_this_goal = 0
                escape_subphase = None
                escape_remaining = 0
                pos_hist.clear()
                wander_mode = False
                consecutive_failed_skips = 0
                print(
                    f"  [{scene_model}] reached segment -> goal_room={next_room!r} "
                    f"goal=({goal_x:.2f},{goal_y:.2f}) path={'ok' if path is not None else 'None'}"
                )

            if robot_below_floor(robot, floor_z):
                fall_recoveries += 1
                print(
                    f"  [{scene_model}] [WARN] Robot below floor at step {step} "
                    f"(z={robot_height_z(robot):.2f}); recovery {fall_recoveries}/{MAX_FALL_RECOVERIES}"
                )
                if fall_recoveries > MAX_FALL_RECOVERIES:
                    print(f"  [{scene_model}] max fall recoveries exceeded, ending rollout")
                    break
                recover_xy = (goal_x, goal_y)
                if not try_land_near_xy(
                    env,
                    robot,
                    floor,
                    recover_xy,
                    interior_margin_m,
                    LAND_NEAR_RADIUS_M,
                    max_tries=200,
                ):
                    if not try_land_near_xy(
                        env,
                        robot,
                        floor,
                        (x, y),
                        interior_margin_m,
                        LAND_NEAR_RADIUS_M,
                        max_tries=200,
                    ):
                        print(f"  [{scene_model}] runtime re-land failed, ending rollout")
                        break
                stabilize_after_land(env, ROBOT_NAME)
                obs, _ = env.get_obs()
                x, y, _ = robot_xy_yaw(robot)
                path, _ = compute_global_path(env, robot, floor, (x, y), (goal_x, goal_y))
                replan_counter = 0
                continue

        ok = True
        print(
            f"  [{scene_model}] wrote {video_path} ({step} frames) "
            f"nav_csv={nav_csv_path}"
        )
    finally:
        if nav_csv_file is not None:
            try:
                nav_csv_file.close()
            except OSError:
                pass
        if writer is not None:
            writer.release()
            finalize_mp4_for_apple_players(video_path)
        del env

    return ok


def _read_scene_names(scenes_file: Path) -> list[str]:
    if not scenes_file.is_file():
        return list(DEFAULT_SCENE_NAMES)
    names: list[str] = []
    for line in scenes_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            names.append(s)
    return names if names else list(DEFAULT_SCENE_NAMES)


def _behavior_scenes_dir(data_path: Path) -> Path:
    return data_path / "behavior-1k-assets" / "scenes"


def _scenes_downloaded(data_path: Path) -> bool:
    p = _behavior_scenes_dir(data_path)
    if not p.is_dir():
        return False
    return any(
        x.is_dir() and not x.name.startswith(".") and x.name != "background" for x in p.iterdir()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render first-person robot navigation videos (OmniGibson).")
    parser.add_argument("--scenes-file", type=Path, default=_REPO_ROOT / "selected_scenes.txt")
    parser.add_argument("--scenes", type=str, default=None, help="Comma-separated scene names.")
    parser.add_argument("--nav-root", type=Path, default=DEFAULT_NAV_ROOT, help="Directory with <scene>/nav_paths.json.")
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "output2" / "robot_videos")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps (default: max-seconds * fps).")
    parser.add_argument("--fps", type=float, default=float(ACTION_FREQ))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--floor", type=int, default=None)
    parser.add_argument("--interior-margin", type=float, default=DEFAULT_INTERIOR_MARGIN_M)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    max_steps = args.max_steps
    if max_steps is None:
        max_steps = int(math.ceil(args.max_seconds * args.fps))

    _prepare_runtime_env(headless=args.headless, data_path=args.data_path)

    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import download_behavior_1k_assets

    data_path = Path(gm.DATA_PATH)
    print(f"OMNIGIBSON_DATA_PATH={gm.DATA_PATH}")

    if not _scenes_downloaded(data_path):
        if args.skip_download:
            sys.exit("Dataset missing and --skip-download set.")
        if not args.accept_license:
            sys.exit("Re-run with --accept-license to download BEHAVIOR-1K assets.")
        download_behavior_1k_assets(accept_license=True)

    if args.scenes:
        scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scene_names = _read_scene_names(args.scenes_file.expanduser().resolve())

    out_root = args.output_dir.expanduser().resolve()
    nav_root = args.nav_root.expanduser().resolve()
    print(f"Scenes: {len(scene_names)}, max_steps={max_steps}, output={out_root}")

    ok = 0
    for i, scene_model in enumerate(scene_names):
        try:
            if run_scene_video(
                scene_model,
                nav_root,
                resolution=args.resolution,
                max_steps=max_steps,
                fps=args.fps,
                output_dir=out_root,
                structure_only=args.structure_only,
                floor_arg=args.floor,
                interior_margin_m=args.interior_margin,
                seed=None if args.seed is None else args.seed + i,
            ):
                ok += 1
        except Exception as e:
            print(f"FAIL {scene_model}: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            if i < len(scene_names) - 1 and og.sim is not None:
                try:
                    og.clear()
                except Exception as ce:
                    print(f"Warning: og.clear() failed: {ce}", file=sys.stderr)

    print(f"\nDone. {ok}/{len(scene_names)} scene(s) produced video(s).")
    og.shutdown()


if __name__ == "__main__":
    main()
