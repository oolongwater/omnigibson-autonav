#!/usr/bin/env python3
"""
Autonomous navigation for 180 continuous simulation seconds in OmniGibson — Benevolence_2_int.

Uses A* on the traversable map (with-objects trav + painted doorways + archway passages) + pure
pursuit on the path, bearing-to-goal when A* fails, 2D LiDAR avoidance, and stuck recovery.
Walls are solid (collision on); door panels are opened then non-collidable so openings are passable.

Run from anywhere; ensures BEHAVIOR-1K/OmniGibson is on sys.path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, TextIO

import cv2
import numpy as np
import torch as th

# Repo layout: OmniGibson_TakeHomeTest/BEHAVIOR-1K/OmniGibson/omnigibson
_REPO_ROOT = Path(__file__).resolve().parent
_OG_SRC = _REPO_ROOT / "BEHAVIOR-1K" / "OmniGibson"
if _OG_SRC.is_dir() and str(_OG_SRC) not in sys.path:
    sys.path.insert(0, str(_OG_SRC))

import omnigibson as og
from omnigibson.object_states import ContactBodies, Open
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.tasks.dummy_task import DummyTask
import omnigibson.utils.transform_utils as T
from omnigibson.utils.sim_utils import land_object, test_valid_pose

# Default sim: 30 Hz control (see omnigibson.macros gm.DEFAULT_SIM_STEP_FREQ)
ACTION_FREQ = 30
SIM_SECONDS = 180
N_STEPS = ACTION_FREQ * SIM_SECONDS  # 5400

# LiDAR (must match turtlebot ScanSensor kwargs in config)
LIDAR_MIN_RANGE = 0.05
LIDAR_MAX_RANGE = 10.0

# Navigation parameters
LOOK_AHEAD_M = 0.75
LOOK_AHEAD_MIN_M = 0.25
REPLAN_EVERY_STEPS = 45
PATH_DEVIATION_REPLAN_M = 0.5
GOAL_REACH_M = 0.40
# When A* path exists and we are this close to goal, do not apply pp_slow (trust planner clearance).
TRUST_ASTAR_NEAR_GOAL_M = 1.0
KP_ANG = 2.2
V_MAX_NORM = 0.85
W_MAX_NORM = 0.9

# Obstacle avoidance (wider sectors / earlier slow for furniture clearance)
FRONT_HALF_DEG = 45.0
SIDE_DEG = 70.0
SLOW_DIST = 0.55  # used for v scaling when slowing near obstacles
SLOW_DIST_ENTER = 0.45  # enter reactive avoidance (hysteresis)
SLOW_DIST_EXIT = 0.62  # exit reactive avoidance (hysteresis)
STOP_DIST = 0.15

# Stuck recovery: if barely moving over a window, skip to next goal
STUCK_WINDOW_STEPS = 60
STUCK_DIST_THRESHOLD_M = 0.20
# Min closing distance toward goal over STUCK_WINDOW_STEPS (replan-invariant; uses dist_goal_hist).
# Slightly below 0.10: ~0.098 progress was misclassified as "stuck" near the sofa corner on EC2.
STUCK_PATH_PROGRESS_THRESHOLD_M = 0.085
# First stuck episode: backup+rotate recovery; second on same goal: skip.
MAX_STUCK_SKIPS_PER_GOAL = 3

# Force recovery when wedged: contact + near-zero cmd for this many steps (~0.5 s at 30 Hz).
CONTACT_STALL_TRIGGER_STEPS = 15
CONTACT_STALL_V_THRESH = 0.08
CONTACT_STALL_W_THRESH = 0.08

# Recovery sub-steps (executed inside main control loop)
RECOVERY_BACKUP_STEPS = 10
RECOVERY_BACKUP_V = -0.5
RECOVERY_ROTATE_MAX_STEPS = 15
RECOVERY_ALIGN_TOL_RAD = math.radians(20.0)

# VFH-style goal-biased obstacle avoidance
VFH_NUM_BINS = 25  # odd => one bin centered at 0° in ±120° sector
VFH_FRONT_SECTOR_RAD = math.radians(120.0)  # ±120° in lidar frame (0 = forward)
VFH_CLEARANCE_MARGIN_M = 0.12  # on top of STOP_DIST for bin to be "free"
VFH_PP_BEAM_HALF_RAD = math.radians(10.0)  # Candidate A: clear along PP/bearing heading

# Steering smoothing (normalized w in [-1,1] action space)
W_CMD_DEADBAND = 0.03
W_CMD_EMA_NEW = 0.7  # w = W_CMD_EMA_NEW * w_new + (1-W_CMD_EMA_NEW) * w_prev

# Paint furniture collision AABBs into trav map (dataset map omits many objects)
FURNITURE_CATEGORIES = [
    "bottom_cabinet",
    "top_cabinet",
    "breakfast_table",
    "sofa",
    "bed",
    "shelf",
    "fridge",
    "stove",
    "countertop",
    "toilet",
    "bathtub",
]
OBSTACLE_INFLATE_CELLS = 1
# Extra per-category AABB dilation (cells); 1 cell = 0.10 m at trav_map_resolution 0.1. Others use OBSTACLE_INFLATE_CELLS.
FURNITURE_INFLATE_CELLS: dict[str, int] = {
    "sofa": 3,  # ~0.30 m buffer at 0.1 m/cell (plan)
}

# Optional trav-map blocker rects in world meters (x0, y0, x1, y1), painted blocked
# after paint_object_obstacles. Empty: use direct closet_0<->corridor_0 (see nav_paths).
EXTRA_TRAV_BLOCKERS_M: list[tuple[float, float, float, float]] = []

# Snap search around each PDF waypoint; max offset from original (m) to avoid silent cross-room snaps.
SNAP_SEARCH_RADIUS_M = 2.0
SNAP_MAX_OFFSET_M = 1.2

# Diagnostic output (under output/)
NAV_CSV_NAME = "autonomous_nav_benevolence2_nav.csv"
EVENTS_LOG_NAME = "autonomous_nav_benevolence2_events.log"
SUMMARY_JSON_NAME = "autonomous_nav_benevolence2_summary.json"
CSV_FLUSH_EVERY = 30

# Stabilization after landing
POST_LAND_SIM_STEPS = 15
POST_LAND_RENDER_PASSES = 3

# Goal marker appearance (matches PointNavigationTask defaults)
MARKER_RADIUS = 0.36
MARKER_HEIGHT = 0.01

# Preset waypoints (world x, y, label) for Benevolence_2_int — 7 marked Xs.
# x1-x3,x7 from nav_paths centroids; x4 on closet_0->corridor_0 polyline; x5 shifted west;
# x6 bedroom_1 centroid south of bed (nav_paths); avoids closet_1->bedroom_1 cutting through bed.
# Ordered tour: x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 (see output/benevolence2_marked_path.png).
# Spawn = x1. Runtime snap_to_reachable() may adjust each goal slightly per chain step.
PRESET_GOALS: list[tuple[float, float, str]] = [
    (0.4523,  6.0895, "marked_X1_bathroom_0"),
    (3.0923,  5.602,  "marked_X2_bedroom_0"),
    (3.0923,  3.737,  "marked_X3_closet_0"),
    # Corridor pinch exit (nav_paths closet_0->corridor_0); (-0.30,2.33) failed snap.
    (0.52,    2.33,   "marked_X4_corridor_0"),
    (0.10,   -2.15,   "marked_X5_closet_1"),
    (2.85,   -1.06,   "marked_X6_bedroom_1"),
    (3.0923,  2.2295, "marked_X7_bathroom_1"),
]

# Fallback rectangles (row0, row1, col0, col1) if no door objects are in the scene registry.
DOORWAYS: list[tuple[int, int, int, int]] = []

# Half-width in trav-map cells when painting door + passage openings (clipped to wall-free mask).
DOORWAY_RADIUS_CELLS = 12

# Open archways (via_doors=[] in nav_paths.json: closet_0<->corridor_0, closet_1<->corridor_0, etc.)
# need explicit trav-map paint at runtime — door objects don't exist for these passages.
# main() additionally loads output/scene_graphs/Benevolence_2_int/nav_paths.json and appends every
# inter-room waypoint into this list before prepare_trav_map_doorways runs.
PASSAGE_WAYPOINTS: list[tuple[float, float]] = []
NAV_PATHS_JSON_PATH = (
    _REPO_ROOT / "output" / "scene_graphs" / "Benevolence_2_int" / "nav_paths.json"
)


def load_passage_waypoints_from_nav_paths(json_path: Path) -> list[tuple[float, float]]:
    """Return every internal waypoint from every path in nav_paths.json (deduped to ~10 cm grid)."""
    if not json_path.is_file():
        og.log.warning(
            f"Nav-paths JSON not found at {json_path}; passage carving will rely on door objects only."
        )
        return []
    try:
        blob = json.loads(json_path.read_text())
    except (OSError, ValueError) as e:
        og.log.warning(f"Could not parse {json_path}: {e}")
        return []
    seen: set[tuple[int, int]] = set()
    out: list[tuple[float, float]] = []
    for entry in blob.get("paths", []):
        wps = entry.get("waypoints") or []
        # Drop the first/last (room centroids); keep only the interior pinch points.
        for p in wps[1:-1]:
            try:
                x = float(p[0])
                y = float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            key = (int(round(x * 10.0)), int(round(y * 10.0)))
            if key in seen:
                continue
            seen.add(key)
            out.append((x, y))
    return out

# Spawn candidates: anchor at x1 (bathroom_0) using nav_paths waypoints into bathroom_0 so the chain
# x1→x2→…→x7 is reachable. spawn_near_goals requires distance >= GOAL_REACH_M + 0.25 = 0.65 m to x1.
SPAWN_CANDIDATES: list[tuple[float, float]] = [
    (1.25, 5.60),   # bathroom_0->bedroom_0 waypoint, ~1.05 m east of x1
    (1.10, 5.87),   # ~0.71 m east of x1
    (0.77, 6.08),   # nav_paths waypoint, ~0.32 m east of x1 (relaxed fallback)
    (0.45, 6.09),   # x1 itself (final fallback)
]


@dataclass
class RunDiagnostics:
    """Counters and samples for post-run summary (JSON + stdout)."""

    replan_count: int = 0
    astar_fail_count: int = 0
    vfh_override_steps: int = 0
    pp_slow_steps: int = 0
    recovery_episodes: int = 0
    contact_stall_triggers: int = 0
    d_front_samples: list[float] = field(default_factory=list)
    events_fp: Optional[TextIO] = None

    def log_event(self, kind: str, **fields: Any) -> None:
        if self.events_fp is None:
            return
        rec = {"kind": kind, **fields}
        self.events_fp.write(json.dumps(rec, default=str) + "\n")
        self.events_fp.flush()

    def record_d_front(self, d: float) -> None:
        self.d_front_samples.append(float(d))


@dataclass
class VfhBlendCommand:
    """Reactive avoidance output when VFH hysteresis is active."""

    mode: Literal["pp_slow", "vfh", "emergency"]
    v_scale: float = 1.0
    v: float = 0.0
    w: float = 0.0


class VisualizedDummyTask(DummyTask):
    """DummyTask with visible goal (blue) and start (red) cylinder markers on the floor."""

    def _load(self, env):
        super()._load(env)
        self._goal_marker = PrimitiveObject(
            relative_prim_path="/nav_goal_marker",
            primitive_type="Cylinder",
            name="nav_goal_marker",
            radius=MARKER_RADIUS,
            height=MARKER_HEIGHT,
            visual_only=True,
            rgba=th.tensor([0.0, 0.0, 1.0, 0.3]),
        )
        self._start_marker = PrimitiveObject(
            relative_prim_path="/nav_start_marker",
            primitive_type="Cylinder",
            name="nav_start_marker",
            radius=MARKER_RADIUS,
            height=MARKER_HEIGHT,
            visual_only=True,
            rgba=th.tensor([1.0, 0.0, 0.0, 0.3]),
        )
        env.scene.add_object(self._goal_marker)
        env.scene.add_object(self._start_marker)

        og.sim.play()
        self._goal_marker.set_position_orientation(position=th.tensor([0.0, 0.0, 100.0]))
        self._start_marker.set_position_orientation(position=th.tensor([0.0, 0.0, 100.0]))
        env.scene.update_initial_file()
        og.sim.stop()

    def set_goal_marker(self, x: float, y: float, z: float):
        self._goal_marker.set_position_orientation(position=th.tensor([x, y, z]))

    def set_start_marker(self, x: float, y: float, z: float):
        self._start_marker.set_position_orientation(position=th.tensor([x, y, z]))


def build_env_config() -> dict:
    """Fork of turtlebot_nav.yaml: walls + furniture; trav with objects + doorway paint; doors opened at runtime."""
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
            "viewer_width": 1280,
            "viewer_height": 720,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Benevolence_2_int",
            "trav_map_resolution": 0.1,
            "default_erosion_radius": 0.0,
            # With-objects map marks walls; painted DOORWAYS + door positions restore traversable graph through openings.
            "trav_map_with_objects": True,
            "num_waypoints": 1,
            "waypoint_resolution": 0.2,
            "load_object_categories": [
                "floors",
                "ceilings",
                "walls",
                "door",
                "sliding_door",
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
            ],
            "not_load_object_categories": None,
            "load_room_types": None,
            "load_room_instances": None,
            "seg_map_resolution": 0.1,
            "scene_source": "OG",
            "include_robots": True,
        },
        "robots": [
            {
                "type": "Turtlebot",
                "name": "nav_turtlebot",
                "obs_modalities": ["scan", "rgb", "depth"],
                "include_sensor_names": None,
                "exclude_sensor_names": None,
                "scale": 1.0,
                "self_collision": False,
                "action_normalize": True,
                "action_type": "continuous",
                "sensor_config": {
                    "VisionSensor": {"sensor_kwargs": {"image_height": 256, "image_width": 256}},
                    "ScanSensor": {
                        "sensor_kwargs": {
                            "min_range": LIDAR_MIN_RANGE,
                            "max_range": LIDAR_MAX_RANGE,
                        }
                    },
                },
                "controller_config": {"base": {"name": "DifferentialDriveController"}},
            }
        ],
        "objects": [],
        "task": {"type": "VisualizedDummyTask"},
        "wrapper": {"type": None},
    }


def extract_scan_from_obs(robot_obs: dict) -> th.Tensor | None:
    """Find 1D normalized scan [0,1] from nested robot observation dict."""
    for _sensor_name, sensor_obs in robot_obs.items():
        if isinstance(sensor_obs, dict) and "scan" in sensor_obs:
            s = sensor_obs["scan"]
            if isinstance(s, th.Tensor):
                return s.squeeze().flatten().float()
            return th.as_tensor(s, dtype=th.float32).squeeze().flatten()
    return None


def normalized_scan_to_meters(scan_norm: th.Tensor) -> th.Tensor:
    return scan_norm * (LIDAR_MAX_RANGE - LIDAR_MIN_RANGE) + LIDAR_MIN_RANGE


# --- FPV video recording helpers ---


def extract_rgb_from_obs(robot_obs: dict):
    """Walk nested robot obs dict and return the first ``"rgb"`` tensor found."""
    for _sensor_name, sensor_obs in robot_obs.items():
        if isinstance(sensor_obs, dict) and "rgb" in sensor_obs:
            return sensor_obs["rgb"]
    return None


def rgb_to_bgr_uint8(rgb_tensor) -> np.ndarray:
    if isinstance(rgb_tensor, th.Tensor):
        arr = rgb_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(rgb_tensor)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected HxWx>=3 RGB, got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _resolve_ffmpeg_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for d in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        p = Path(d) / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def finalize_mp4(path: Path) -> None:
    """Re-encode mp4v to H.264 Baseline for QuickTime / browser / IDE preview."""
    if path.suffix.lower() != ".mp4" or not path.is_file():
        return
    ffmpeg_bin = _resolve_ffmpeg_tool("ffmpeg")
    if ffmpeg_bin is None:
        print(f"  [WARN] ffmpeg not found; {path.name} stays mp4v (may not play in QuickTime).")
        return
    tmp = path.with_suffix(".h264.tmp.mp4")
    vf: list[str] = []
    ffprobe_bin = _resolve_ffmpeg_tool("ffprobe")
    if ffprobe_bin:
        try:
            pr = subprocess.run(
                [ffprobe_bin, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
                check=False, capture_output=True, text=True,
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
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
        "-i", str(path), "-shortest",
        "-map", "1:v:0", "-map", "0:a:0",
        *vf,
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
        "-crf", "18", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "48k", "-ac", "1",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        tmp.replace(path)
        if sys.platform == "darwin":
            subprocess.run(["xattr", "-c", str(path)], check=False, capture_output=True)
        print(f"  Finalized {path.name} -> H.264")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"  [WARN] ffmpeg finalize failed for {path}: {e}")
        if tmp.is_file():
            tmp.unlink(missing_ok=True)


def wrap_angle_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def robot_xy_yaw(robot) -> tuple[float, float, float]:
    pos, quat = robot.get_position_orientation()
    x = float(pos[0].item())
    y = float(pos[1].item())
    yaw = float(T.z_angle_from_quat(quat).item())
    return x, y, yaw


def distance_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def compute_global_path(env, robot, floor: int, start_xy: tuple[float, float], goal_xy: tuple[float, float]):
    """Returns (path tensor (N,2) or None, geodesic or None)."""
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
    g = float(geo.item()) if isinstance(geo, th.Tensor) else (float(geo) if geo is not None else None)
    return path, g


def validate_preset_goals(
    env, robot, floor: int, start_xy: tuple[float, float]
) -> list[tuple[float, float, str]]:
    """Return preset goals that have an A* path from start_xy; log each check."""
    active: list[tuple[float, float, str]] = []
    for gx, gy, room in PRESET_GOALS:
        path, _ = compute_global_path(env, robot, floor, start_xy, (gx, gy))
        if path is not None:
            print(f"[GOAL_CHECK] OK   {room} ({gx:.2f},{gy:.2f})")
            active.append((gx, gy, room))
        else:
            print(f"[GOAL_CHECK] FAIL {room} ({gx:.2f},{gy:.2f})")
    return active


def snap_to_reachable(
    env,
    robot,
    floor: int,
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
    radius_m: float = SNAP_SEARCH_RADIUS_M,
    step_m: float = 0.10,
    max_offset_m: float = SNAP_MAX_OFFSET_M,
) -> tuple[float, float] | None:
    """
    If A* from start_xy to target fails, search a grid of offsets (sorted by distance to target)
    and return the first candidate with a valid path. Candidates farther than max_offset_m from the
    original target are skipped (prevents snapping X2 to a cell next to X1).
    """
    tx, ty = target_xy
    path, _ = compute_global_path(env, robot, floor, start_xy, (tx, ty))
    if path is not None:
        return (tx, ty)
    steps = max(1, int(round(radius_m / step_m)))
    cand: list[tuple[float, float, float]] = []
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            dx = i * step_m
            dy = j * step_m
            d_off = math.hypot(dx, dy)
            if d_off > max_offset_m + 1e-9:
                continue
            cand.append((tx + dx, ty + dy, d_off))
    cand.sort(key=lambda t: t[2])
    for gx, gy, d_off in cand:
        if d_off > max_offset_m + 1e-9:
            break
        path2, _ = compute_global_path(env, robot, floor, start_xy, (gx, gy))
        if path2 is not None:
            return (gx, gy)
    return None


def build_snapped_goal_tour(
    env,
    robot,
    floor: int,
    spawn_xy: tuple[float, float],
    snap_radius_m: float = SNAP_SEARCH_RADIUS_M,
    max_offset_m: float = SNAP_MAX_OFFSET_M,
) -> tuple[list[tuple[float, float, str]], list[dict[str, Any]]]:
    """
    Snap each PRESET_GOAL in order: try A* from the previous snapped waypoint, else from spawn.
    Drops goals that cannot be snapped from either start.
    """
    snapped: list[tuple[float, float, str]] = []
    meta: list[dict[str, Any]] = []
    prev = spawn_xy
    for gx, gy, room in PRESET_GOALS:
        res = snap_to_reachable(
            env, robot, floor, prev, (gx, gy), radius_m=snap_radius_m, max_offset_m=max_offset_m
        )
        snap_from = prev
        if res is None and math.hypot(prev[0] - spawn_xy[0], prev[1] - spawn_xy[1]) > 1e-3:
            res = snap_to_reachable(
                env, robot, floor, spawn_xy, (gx, gy), radius_m=snap_radius_m, max_offset_m=max_offset_m
            )
            snap_from = spawn_xy
        if res is None:
            print(
                f"[SNAP] FAIL {room} ({gx:.2f},{gy:.2f}) — no reachable cell within "
                f"{snap_radius_m:.1f} m, max_offset {max_offset_m:.1f} m"
            )
            meta.append(
                {
                    "room": room,
                    "original_xy": [gx, gy],
                    "snapped_xy": None,
                    "from_xy": [prev[0], prev[1]],
                    "status": "fail",
                }
            )
            continue
        nx, ny = res
        if abs(nx - gx) > 1e-4 or abs(ny - gy) > 1e-4:
            print(f"[SNAP] {room} ({gx:.2f},{gy:.2f}) -> ({nx:.2f},{ny:.2f})")
        else:
            print(f"[SNAP] OK   {room} ({gx:.2f},{gy:.2f})")
        snapped.append((nx, ny, room))
        meta.append(
            {
                "room": room,
                "original_xy": [gx, gy],
                "snapped_xy": [nx, ny],
                "from_xy": [snap_from[0], snap_from[1]],
                "snap_from_spawn": snap_from == spawn_xy,
                "status": "ok",
            }
        )
        prev = (nx, ny)
    return snapped, meta


def run_nav_checks(
    env,
    robot,
    floor: int,
    spawn_xy: tuple[float, float],
    active_goals: list[tuple[float, float, str]],
) -> list[dict[str, Any]]:
    """Verify A* along spawn -> each goal -> loop back to first goal; prints [NAVCHECK] lines."""
    checks: list[dict[str, Any]] = []
    if not active_goals:
        return checks
    pts: list[tuple[float, float]] = [(spawn_xy[0], spawn_xy[1])]
    pts.extend((gx, gy) for gx, gy, _ in active_goals)
    labels = ["spawn"] + [r for _, _, r in active_goals]
    for i in range(len(active_goals)):
        a = (pts[i][0], pts[i][1])
        b = (pts[i + 1][0], pts[i + 1][1])
        la, lb = labels[i], labels[i + 1]
        path, geo = compute_global_path(env, robot, floor, a, b)
        ok = path is not None
        g_m: float | None = None
        if ok and geo is not None:
            g_m = float(geo.item()) if isinstance(geo, th.Tensor) else float(geo)
        suffix = f" len={g_m:.2f}m" if g_m is not None else ""
        print(f"[NAVCHECK] {la} -> {lb} {'OK' if ok else 'FAIL'}{suffix}")
        checks.append(
            {
                "from_label": la,
                "to_label": lb,
                "ok": ok,
                "geodesic_m": g_m,
                "from_xy": [a[0], a[1]],
                "to_xy": [b[0], b[1]],
            }
        )
    # Loop: last goal -> first goal (matches cycling goal_idx in main loop)
    a = (pts[-1][0], pts[-1][1])
    b = (pts[1][0], pts[1][1])
    la, lb = labels[-1], labels[1]
    path, geo = compute_global_path(env, robot, floor, a, b)
    ok = path is not None
    g_m = None
    if ok and geo is not None:
        g_m = float(geo.item()) if isinstance(geo, th.Tensor) else float(geo)
    suffix = f" len={g_m:.2f}m" if g_m is not None else ""
    print(f"[NAVCHECK] {la} -> {lb} {'OK' if ok else 'FAIL'}{suffix}")
    checks.append(
        {
            "from_label": la,
            "to_label": lb,
            "ok": ok,
            "geodesic_m": g_m,
            "from_xy": [a[0], a[1]],
            "to_xy": [b[0], b[1]],
        }
    )
    return checks


def first_reachable_from_index(
    env,
    robot,
    floor: int,
    start_xy: tuple[float, float],
    active_goals: list[tuple[float, float, str]],
    start_index: int,
) -> tuple[float, float, str | None, th.Tensor | None, int, float | None]:
    """Try active_goals[start_index], then wrap, until one has a path."""
    n = len(active_goals)
    if n == 0:
        return 0.0, 0.0, None, None, 0, None
    for k in range(n):
        j = (start_index + k) % n
        gx, gy, room = active_goals[j]
        path, geo = compute_global_path(env, robot, floor, start_xy, (gx, gy))
        if path is not None:
            g = float(geo.item()) if isinstance(geo, th.Tensor) else (float(geo) if geo is not None else None)
            return gx, gy, room, path, j, g
    gx, gy, room = active_goals[start_index % n]
    return gx, gy, room, None, start_index % n, None


def trav_floor_z(env, floor: int = 0) -> float:
    """Floor height from traversability map (Scene.get_floor_height is 0.0 for InteractiveTraversableScene)."""
    h = env.scene.trav_map.floor_heights[floor]
    return float(h.item()) if isinstance(h, th.Tensor) else float(h)


def spawn_near_goals(env, robot, floor: int, floor_z: float):
    """
    Land the robot on a pose that shares a traversable component with at least one PRESET_GOAL
    (avoids random spawn on disconnected islands before doorways are opened in trav map).
    Prefer a spawn that reaches PRESET_GOALS[0] first so the snapped tour can chain X1→…→X7.
    """
    z_off = env.initial_pos_z_offset
    quat = T.euler2quat(th.tensor([0.0, 0.0, 0.0]))
    gx0, gy0, _ = PRESET_GOALS[0]
    # Avoid landing on/near the first waypoint: prev_dist_to_goal starts at +inf, so the first sim step
    # would otherwise edge-trigger "goal reached" while still inside GOAL_REACH_M and --once exits with
    # a trivial clip.
    min_spawn_to_first = GOAL_REACH_M + 0.25
    for sx, sy in SPAWN_CANDIDATES:
        pos = th.tensor([sx, sy, floor_z], dtype=th.float32)
        if not test_valid_pose(robot, pos, quat, z_off):
            continue
        path, _ = compute_global_path(env, robot, floor, (sx, sy), (gx0, gy0))
        if path is None:
            continue
        if distance_xy(sx, sy, gx0, gy0) >= min_spawn_to_first:
            land_object(robot, pos, quat, z_off)
            return pos, quat
    for sx, sy in SPAWN_CANDIDATES:
        pos = th.tensor([sx, sy, floor_z], dtype=th.float32)
        if not test_valid_pose(robot, pos, quat, z_off):
            continue
        path, _ = compute_global_path(env, robot, floor, (sx, sy), (gx0, gy0))
        if path is not None:
            land_object(robot, pos, quat, z_off)
            return pos, quat
    for sx, sy in SPAWN_CANDIDATES:
        pos = th.tensor([sx, sy, floor_z], dtype=th.float32)
        if not test_valid_pose(robot, pos, quat, z_off):
            continue
        for gx, gy, _ in PRESET_GOALS:
            path, _ = compute_global_path(env, robot, floor, (sx, sy), (gx, gy))
            if path is not None:
                land_object(robot, pos, quat, z_off)
                return pos, quat
    for _ in range(50):
        _, rpos = env.scene.get_random_point(floor=0, robot=robot)
        yaw = (th.rand(1) * math.pi * 2.0).item()
        rquat = T.euler2quat(th.tensor([0.0, 0.0, yaw]))
        if not test_valid_pose(robot, rpos, rquat, z_off):
            continue
        rx, ry = float(rpos[0].item()), float(rpos[1].item())
        for gx, gy, _ in PRESET_GOALS:
            path, _ = compute_global_path(env, robot, floor, (rx, ry), (gx, gy))
            if path is not None:
                land_object(robot, rpos, rquat, z_off)
                return rpos, rquat
    sx, sy = SPAWN_CANDIDATES[0]
    pos = th.tensor([sx, sy, floor_z], dtype=th.float32)
    og.log.warning("spawn_near_goals: no A*-reachable pose found; landing at first candidate.")
    land_object(robot, pos, quat, z_off)
    return pos, quat


def stabilize_after_land(env, robot_name: str):
    """Let physics settle after landing using zero drive commands."""
    zero = th.tensor([0.0, 0.0], dtype=th.float32)
    for _ in range(POST_LAND_SIM_STEPS):
        env.step({robot_name: zero})
    for _ in range(POST_LAND_RENDER_PASSES):
        og.sim.render()


def closest_path_index(path: th.Tensor, x: float, y: float) -> int:
    """Index of waypoint on path closest to (x,y)."""
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


def path_arclength_to_closest(path: th.Tensor, x: float, y: float) -> float:
    """Cumulative polyline length from path[0] to path[closest_index] (meters)."""
    if path is None or path.numel() == 0:
        return 0.0
    i = closest_path_index(path, x, y)
    s = 0.0
    for k in range(i):
        s += distance_xy(
            float(path[k, 0].item()),
            float(path[k, 1].item()),
            float(path[k + 1, 0].item()),
            float(path[k + 1, 1].item()),
        )
    return s


def effective_lookahead_m(d_front: float | None) -> float:
    """Shorten PP lookahead in tight spaces using LiDAR front clearance."""
    if d_front is None:
        return LOOK_AHEAD_M
    return min(LOOK_AHEAD_M, max(LOOK_AHEAD_MIN_M, 0.8 * d_front))


def lookahead_point(path: th.Tensor, x: float, y: float, look_ahead: float) -> tuple[float, float] | None:
    """Advance ~look_ahead m along the path polyline from the closest waypoint (pure pursuit)."""
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
    d_front: float | None = None,
) -> tuple[float, float]:
    """Returns normalized (lin_vel, ang_vel) in [-1,1]."""
    look = effective_lookahead_m(d_front)
    target = lookahead_point(path, x, y, look)
    if target is None:
        return 0.0, 0.0
    tx, ty = target
    heading = math.atan2(ty - y, tx - x)
    err = wrap_angle_pi(heading - yaw)
    w = max(-1.0, min(1.0, KP_ANG * err / math.pi))
    align = max(0.0, math.cos(err))
    v = V_MAX_NORM * align
    if abs(err) > math.radians(75):
        v *= 0.25
    return max(-1.0, min(1.0, v)), max(-1.0, min(1.0, w))


def bearing_to_goal_cmd(x: float, y: float, yaw: float, gx: float, gy: float) -> tuple[float, float]:
    """Turn-on-the-spot until aligned, then drive forward (avoids v≈0 while still misaligned in doorways)."""
    heading = math.atan2(gy - y, gx - x)
    err = wrap_angle_pi(heading - yaw)
    w = max(-1.0, min(1.0, KP_ANG * err / math.pi))
    if abs(err) > math.radians(22):
        return 0.0, w
    return V_MAX_NORM, 0.0


def pure_pursuit_desired_heading_rel(
    x: float,
    y: float,
    yaw: float,
    path: th.Tensor,
    d_front: float | None = None,
) -> float | None:
    """Angle (rad) from robot forward to lookahead point; None if no path segment."""
    look = effective_lookahead_m(d_front)
    target = lookahead_point(path, x, y, look)
    if target is None:
        return None
    tx, ty = target
    heading = math.atan2(ty - y, tx - x)
    return wrap_angle_pi(heading - yaw)


def bearing_goal_desired_heading_rel(x: float, y: float, yaw: float, gx: float, gy: float) -> float:
    heading = math.atan2(gy - y, gx - x)
    return wrap_angle_pi(heading - yaw)


def lidar_sector_mins(dist_m: th.Tensor) -> tuple[float, float, float, float]:
    """Min range in front / left / right sectors and global min (meters)."""
    n = dist_m.shape[0]
    if n < 8:
        z = float(LIDAR_MAX_RANGE)
        return z, z, z, z
    angles = th.linspace(-math.pi, math.pi, n, device=dist_m.device, dtype=dist_m.dtype)
    deg = angles * (180.0 / math.pi)
    front = (deg.abs() <= FRONT_HALF_DEG) | (deg.abs() >= 360.0 - FRONT_HALF_DEG)
    left = (deg > FRONT_HALF_DEG) & (deg < SIDE_DEG)
    right = (deg < -FRONT_HALF_DEG) & (deg > -SIDE_DEG)
    d_front = float(th.min(dist_m[front]).item()) if bool(front.any()) else LIDAR_MAX_RANGE
    d_left = float(th.min(dist_m[left]).item()) if bool(left.any()) else LIDAR_MAX_RANGE
    d_right = float(th.min(dist_m[right]).item()) if bool(right.any()) else LIDAR_MAX_RANGE
    d_min360 = float(th.min(dist_m).item())
    return d_front, d_left, d_right, d_min360


def min_range_in_beam(dist_m: th.Tensor, center_rel_rad: float, half_width_rad: float) -> float:
    """Minimum LiDAR range within ±half_width_rad of center_rel_rad (robot frame)."""
    n = int(dist_m.shape[0])
    if n < 8:
        return float(LIDAR_MAX_RANGE)
    angles = th.linspace(-math.pi, math.pi, n, device=dist_m.device, dtype=dist_m.dtype)
    diff = th.atan2(th.sin(angles - center_rel_rad), th.cos(angles - center_rel_rad))
    mask = diff.abs() <= half_width_rad
    if not bool(mask.any()):
        return float(LIDAR_MAX_RANGE)
    return float(th.min(dist_m[mask]).item())


def _vfh_bin_loop(
    dist_m: th.Tensor,
    desired_heading_rel: float,
    angles: th.Tensor,
    clearance_need: float,
) -> tuple[float, float, float]:
    """Return (best_score, best_center_rad, best_clear_m). best_score < 0 if no clear bin."""
    n_bins = VFH_NUM_BINS
    edges = th.linspace(-VFH_FRONT_SECTOR_RAD, VFH_FRONT_SECTOR_RAD, n_bins + 1, device=dist_m.device)
    best_score = -1.0
    best_center = 0.0
    best_clear = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (angles >= lo) & (angles < hi)
        if i == n_bins - 1:
            in_bin = (angles >= lo) & (angles <= hi)
        if not bool(in_bin.any()):
            d_bin = float(LIDAR_MAX_RANGE)
        else:
            d_bin = float(th.min(dist_m[in_bin]).item())
        if d_bin < clearance_need:
            continue
        center = float(((lo + hi) / 2.0).item())
        align = math.cos(wrap_angle_pi(center - desired_heading_rel))
        if align < 0.0:
            align *= 0.25
        score = d_bin * align
        if score > best_score:
            best_score = score
            best_center = center
            best_clear = d_bin
    return best_score, best_center, best_clear


def vfh_recovery_heading_meta(dist_m: th.Tensor, desired_heading_rel: float) -> dict[str, Any]:
    """Pick a relative yaw target for recovery rotate (best VFH bin or ±90° toward open side)."""
    meta: dict[str, Any] = {
        "d_front": None,
        "chosen_bin_deg": None,
        "chosen_bin_rad": desired_heading_rel,
        "chosen_clearance_m": None,
        "desired_heading_deg": math.degrees(desired_heading_rel),
        "emergency": False,
        "d_min360": None,
    }
    n = int(dist_m.shape[0])
    if n < 8:
        return meta
    d_front, d_left, d_right, d_min360 = lidar_sector_mins(dist_m)
    meta["d_front"] = d_front
    meta["d_min360"] = d_min360
    clearance_need = STOP_DIST + VFH_CLEARANCE_MARGIN_M
    angles = th.linspace(-math.pi, math.pi, n, device=dist_m.device, dtype=dist_m.dtype)
    best_score, best_center, best_clear = _vfh_bin_loop(dist_m, desired_heading_rel, angles, clearance_need)
    meta["chosen_bin_deg"] = math.degrees(best_center) if best_score >= 0.0 else None
    meta["chosen_clearance_m"] = best_clear if best_score >= 0.0 else None
    if best_score < 0.0:
        meta["emergency"] = True
        turn = 1.0 if d_left > d_right else -1.0
        meta["chosen_bin_rad"] = turn * (math.pi / 2.0)
        meta["chosen_bin_deg"] = math.degrees(float(meta["chosen_bin_rad"]))
        return meta
    meta["chosen_bin_rad"] = best_center
    return meta


def vfh_goal_biased_when_active(
    dist_m: th.Tensor,
    desired_heading_rel: float,
    d_front: float,
) -> tuple[VfhBlendCommand, dict[str, Any]]:
    """
    Called only when VFH hysteresis is active (front congestion). Candidate A: PP/bearing beam clear
    → slow along PP (pp_slow). Else bin loop (odd bins, one at 0°). Else emergency turn in place.
    """
    meta: dict[str, Any] = {
        "d_front": d_front,
        "chosen_bin_deg": None,
        "chosen_bin_rad": desired_heading_rel,
        "chosen_clearance_m": None,
        "desired_heading_deg": math.degrees(desired_heading_rel),
        "emergency": False,
        "d_min360": None,
        "d_beam_pp": None,
        "mode": None,
    }
    n = int(dist_m.shape[0])
    if n < 8:
        meta["mode"] = "pp_slow"
        meta["v_scale"] = 1.0
        return VfhBlendCommand(mode="pp_slow", v_scale=1.0), meta

    _df, d_left, d_right, d_min360 = lidar_sector_mins(dist_m)
    meta["d_min360"] = d_min360
    clearance_need = STOP_DIST + VFH_CLEARANCE_MARGIN_M
    d_beam = min_range_in_beam(dist_m, desired_heading_rel, VFH_PP_BEAM_HALF_RAD)
    meta["d_beam_pp"] = d_beam
    if d_beam >= clearance_need:
        v_scale = max(0.0, (d_front - STOP_DIST) / max(SLOW_DIST - STOP_DIST, 1e-6))
        meta["mode"] = "pp_slow"
        meta["v_scale"] = v_scale
        return VfhBlendCommand(mode="pp_slow", v_scale=v_scale), meta

    angles = th.linspace(-math.pi, math.pi, n, device=dist_m.device, dtype=dist_m.dtype)
    best_score, best_center, best_clear = _vfh_bin_loop(dist_m, desired_heading_rel, angles, clearance_need)
    meta["chosen_bin_deg"] = math.degrees(best_center) if best_score >= 0.0 else None
    meta["chosen_clearance_m"] = best_clear if best_score >= 0.0 else None

    if best_score < 0.0:
        meta["emergency"] = True
        meta["mode"] = "emergency"
        turn = 1.0 if d_left > d_right else -1.0
        return VfhBlendCommand(mode="emergency", v=0.0, w=turn * W_MAX_NORM), meta

    meta["chosen_bin_rad"] = best_center
    meta["mode"] = "vfh"
    err = wrap_angle_pi(best_center)
    w_cmd = max(-1.0, min(1.0, KP_ANG * err / math.pi))
    v_scale = max(0.0, (d_front - STOP_DIST) / max(SLOW_DIST - STOP_DIST, 1e-6))
    v_cmd = max(-1.0, min(1.0, V_MAX_NORM * max(0.0, math.cos(err)) * v_scale))
    if d_front < STOP_DIST:
        v_cmd = 0.0
    return VfhBlendCommand(mode="vfh", v=v_cmd, w=w_cmd), meta


def apply_vfh_blend(v_pp: float, w_pp: float, blend: VfhBlendCommand | None) -> tuple[float, float]:
    """Apply pp_slow (scale v only), full vfh/emergency override, or nominal PP/bearing."""
    if blend is None:
        return max(-1.0, min(1.0, v_pp)), max(-1.0, min(1.0, w_pp))
    if blend.mode == "pp_slow":
        return max(-1.0, min(1.0, v_pp * blend.v_scale)), max(-1.0, min(1.0, w_pp))
    return max(-1.0, min(1.0, blend.v)), max(-1.0, min(1.0, blend.w))


def snapshot_wall_free_mask(env, floor: int = 0) -> np.ndarray:
    """
    Boolean mask True = traversable in the wall-only dataset image (floor_trav_no_obj),
    resized to the runtime trav grid. Used to clip doorway/passage paints so we never
    carve free space through walls. If the no-obj PNG is missing, fall back to cells
    that are already free on the loaded floor map (before furniture paint).
    """
    trav = env.scene.trav_map
    fmap = trav.floor_map[floor]
    h, w = int(fmap.shape[0]), int(fmap.shape[1])
    layout = os.path.join(env.scene.scene_dir, "layout")
    png_path = os.path.join(layout, f"floor_trav_no_obj_{floor}.png")
    if os.path.isfile(png_path):
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            return (img >= 250).astype(bool)
    og.log.warning(
        f"snapshot_wall_free_mask: missing or unreadable {png_path}; using current floor_map free cells."
    )
    fm = fmap.detach().cpu().numpy()
    return (fm >= 250).astype(bool)


def _write_floor_map_slice(env, floor: int, arr: np.ndarray) -> None:
    """Write numpy float32 [H,W] back to trav.floor_map[floor] preserving device/dtype."""
    trav = env.scene.trav_map
    fmap = trav.floor_map[floor]
    device = fmap.device
    dtype = fmap.dtype
    trav.floor_map[floor] = th.as_tensor(arr, device=device, dtype=dtype)


def _paint_rect_clipped(
    env,
    floor: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    free_mask: np.ndarray,
) -> None:
    """Set fmap to 255 only where free_mask is True inside the rectangle."""
    trav = env.scene.trav_map
    fmap = trav.floor_map[floor]
    sub_free = free_mask[r0 : r1 + 1, c0 : c1 + 1]
    arr = fmap.detach().cpu().numpy().astype(np.float32).copy()
    block = arr[r0 : r1 + 1, c0 : c1 + 1]
    block[sub_free] = 255.0
    arr[r0 : r1 + 1, c0 : c1 + 1] = block
    _write_floor_map_slice(env, floor, arr)


def _paint_rect_unclipped(
    env,
    floor: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> None:
    """Set all cells in the rectangle to traversable (255).

    Used for doorways centered on loaded door objects: ``floor_trav_no_obj`` often marks the
    closed-door voxel column as *wall*, so `_paint_rect_clipped` would paint **zero** cells and
    leave closet/corridor disconnected in the trav graph (bridge_disconnected_goals then draws a
    line the robot cannot physically follow).
    """
    trav = env.scene.trav_map
    fmap = trav.floor_map[floor]
    arr = fmap.detach().cpu().numpy().astype(np.float32).copy()
    arr[r0 : r1 + 1, c0 : c1 + 1] = 255.0
    _write_floor_map_slice(env, floor, arr)


def open_all_doors(env) -> None:
    """Find all door/sliding_door objects and set them fully open (articulated joints)."""
    doors = list(env.scene.object_registry("category", "door", set()))
    sliding = list(env.scene.object_registry("category", "sliding_door", set()))
    all_doors = doors + sliding
    opened = 0
    for door in all_doors:
        if Open in door.states:
            ok = door.states[Open].set_value(True, fully=True)
            if ok:
                opened += 1
                og.log.info(f"Opened door: {door.name} (category={door.category})")
            else:
                og.log.warning(f"Failed to open door: {door.name}")
        else:
            og.log.info(f"Door {door.name} has no Open state (no joints), skipping.")
    og.log.info(f"Opened {opened}/{len(all_doors)} doors.")
    print(
        f"[DOORS] registry: {len(doors)} door + {len(sliding)} sliding_door = {len(all_doors)}; "
        f"opened {opened}; visual_only next",
        flush=True,
    )
    # Opened door panels still narrow the gap; make doors non-collidable (all links via EntityPrim).
    for door in all_doors:
        try:
            door.visual_only = True
            og.log.info(f"Set visual_only (no collision) on door: {door.name}")
        except Exception as e:
            og.log.warning(f"Could not set visual_only on door {door.name}: {e}")


def paint_doorways_from_doors(env, floor: int = 0, free_mask: np.ndarray | None = None) -> int:
    """Paint trav-map rectangles centered on each door's world position.

    Uses **unclipped** rectangles: the wall-only mask from ``floor_trav_no_obj`` often labels the
    door column as blocked, so clipping would paint nothing and leave A* disconnected from physics.
    ``free_mask`` is kept only for API compatibility with ``prepare_trav_map_doorways``.
    """
    del free_mask  # unused; see docstring
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
        _paint_rect_unclipped(env, floor, r0, r1, c0, c1)
        og.log.info(f"Painted doorway for {door.name} at map[{r0}:{r1},{c0}:{c1}] (unclipped)")
        count += 1
    og.log.info(f"Painted {count} doorway(s) in trav map from door positions.")
    print(f"[DOORS] painted {count} doorway rect(s) (unclipped at door centroids)", flush=True)
    return count


def paint_hardcoded_doorways(env, floor: int = 0, free_mask: np.ndarray | None = None) -> None:
    """Paint static DOORWAYS rectangles as traversable (clipped to wall-free mask)."""
    map_sz = int(env.scene.trav_map.floor_map[floor].shape[0])
    if free_mask is None:
        free_mask = np.ones((map_sz, map_sz), dtype=bool)
    for r0, r1, c0, c1 in DOORWAYS:
        _paint_rect_clipped(env, floor, r0, r1, c0, c1, free_mask)
    og.log.info(f"Painted {len(DOORWAYS)} hardcoded doorway rectangle(s) in trav map (clipped).")


def paint_passages(env, floor: int = 0, free_mask: np.ndarray | None = None) -> None:
    """Paint archway passages so A* connects rooms; clipped so paints never open wall or furniture cells."""
    trav = env.scene.trav_map
    map_sz = int(trav.floor_map[floor].shape[0])
    if free_mask is None:
        free_mask = np.ones((map_sz, map_sz), dtype=bool)
    for wx, wy in PASSAGE_WAYPOINTS:
        xy = th.tensor([float(wx), float(wy)])
        rc = trav.world_to_map(xy)
        r, c = int(rc[0].item()), int(rc[1].item())
        r0 = max(0, r - DOORWAY_RADIUS_CELLS)
        r1 = min(map_sz - 1, r + DOORWAY_RADIUS_CELLS)
        c0 = max(0, c - DOORWAY_RADIUS_CELLS)
        c1 = min(map_sz - 1, c + DOORWAY_RADIUS_CELLS)
        _paint_rect_clipped(env, floor, r0, r1, c0, c1, free_mask)
        og.log.info(f"Painted passage archway at world ({wx:.2f},{wy:.2f}) map[{r0}:{r1},{c0}:{c1}] (clipped)")
    og.log.info(f"Painted {len(PASSAGE_WAYPOINTS)} passage waypoint(s) in trav map.")


def paint_object_obstacles(env, floor: int = 0, inflate_cells: int = OBSTACLE_INFLATE_CELLS) -> int:
    """
    Mark furniture collision AABBs as non-traversable on the trav map grid.
    Run after clipped doorway/passage paints so furniture remains blocking (paints cannot erase it).
    Per-category inflation: FURNITURE_INFLATE_CELLS.get(cat, inflate_cells) cells after building each category mask.
    """
    trav = env.scene.trav_map
    fmap = trav.floor_map[floor]
    device = fmap.device
    dtype = fmap.dtype
    map_sz = int(fmap.shape[0])
    combined = np.zeros((map_sz, map_sz), dtype=np.uint8)
    n_regions = 0
    for cat in FURNITURE_CATEGORIES:
        mask_cat = np.zeros((map_sz, map_sz), dtype=np.uint8)
        for obj in env.scene.object_registry("category", cat, set()):
            try:
                aabb_lo, aabb_hi = obj.aabb
            except Exception as e:
                og.log.warning(f"paint_object_obstacles: skip {cat} object (no aabb): {e}")
                continue
            xmin = float(min(aabb_lo[0].item(), aabb_hi[0].item()))
            xmax = float(max(aabb_lo[0].item(), aabb_hi[0].item()))
            ymin = float(min(aabb_lo[1].item(), aabb_hi[1].item()))
            ymax = float(max(aabb_lo[1].item(), aabb_hi[1].item()))
            corners_xy = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
            rs: list[int] = []
            cs: list[int] = []
            for wx, wy in corners_xy:
                rc = trav.world_to_map(th.tensor([wx, wy], dtype=th.float32))
                rs.append(int(rc[0].item()))
                cs.append(int(rc[1].item()))
            r0, r1 = max(0, min(rs)), min(map_sz - 1, max(rs))
            c0, c1 = max(0, min(cs)), min(map_sz - 1, max(cs))
            if r1 >= r0 and c1 >= c0:
                mask_cat[r0 : r1 + 1, c0 : c1 + 1] = 255
                n_regions += 1
        ic = int(FURNITURE_INFLATE_CELLS.get(cat, inflate_cells))
        if ic > 0:
            k = 2 * ic + 1
            mask_cat = cv2.dilate(mask_cat, np.ones((k, k), np.uint8))
        combined = np.maximum(combined, mask_cat)
    arr = fmap.detach().cpu().numpy().astype(np.float32).copy()
    arr[combined.astype(bool)] = 0.0
    trav.floor_map[floor] = th.as_tensor(arr, device=device, dtype=dtype)
    og.log.info(
        f"paint_object_obstacles: {n_regions} AABB region(s) marked blocked "
        f"(default_inflate={inflate_cells}, per_cat={FURNITURE_INFLATE_CELLS})."
    )
    return n_regions


def paint_extra_trav_blockers(env, floor: int = 0) -> int:
    """Paint world-frame rectangles as fully blocked on trav map (after furniture AABB paint).

    Writes in-place into ``trav.floor_map[floor]`` so the same tensor reference kept by TraversableMap
    is updated (some OG builds behave better than replacing the list slot with a new tensor).
    """
    trav = env.scene.trav_map
    fmap = trav.floor_map[floor]
    map_sz = int(fmap.shape[0])
    n = 0
    ncells = 0
    for x0, y0, x1, y1 in EXTRA_TRAV_BLOCKERS_M:
        rc_a = trav.world_to_map(th.tensor([float(x0), float(y0)], dtype=th.float32))
        rc_b = trav.world_to_map(th.tensor([float(x1), float(y1)], dtype=th.float32))
        r0 = max(0, min(int(rc_a[0].item()), int(rc_b[0].item())))
        r1 = min(map_sz - 1, max(int(rc_a[0].item()), int(rc_b[0].item())))
        c0 = max(0, min(int(rc_a[1].item()), int(rc_b[1].item())))
        c1 = min(map_sz - 1, max(int(rc_a[1].item()), int(rc_b[1].item())))
        if r1 >= r0 and c1 >= c0:
            ncells += (r1 - r0 + 1) * (c1 - c0 + 1)
            fmap[r0 : r1 + 1, c0 : c1 + 1] = 0.0
            og.log.info(
                f"paint_extra_trav_blockers: rect world=({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f}) "
                f"map=[{r0}:{r1},{c0}:{c1}] cells={(r1 - r0 + 1) * (c1 - c0 + 1)}"
            )
            n += 1
    print(
        f"[paint_extra_trav_blockers] painted {n} rect(s), total_blocked_cells={ncells} map_sz={map_sz}",
        flush=True,
    )
    return n


def prepare_trav_map_doorways(env, floor: int = 0) -> None:
    """Clip door/passage re-opens to wall-only free space, then paint furniture AABBs last."""
    free_mask = snapshot_wall_free_mask(env, floor=floor)
    n = paint_doorways_from_doors(env, floor=floor, free_mask=free_mask)
    if n == 0:
        og.log.warning("No door objects found in registry; trav map uses hardcoded DOORWAYS only.")
    paint_hardcoded_doorways(env, floor=floor, free_mask=free_mask)
    paint_passages(env, floor=floor, free_mask=free_mask)
    paint_object_obstacles(env, floor=floor)
    # paint_extra_trav_blockers: called after bridge_disconnected_goals in main() so in-place writes
    # apply to the final floor_map tensor and cannot be shadowed by bridge line-paints.


def _world_to_rc(trav, wx: float, wy: float) -> tuple[int, int]:
    rc = trav.world_to_map(th.tensor([wx, wy], dtype=th.float32))
    return int(rc[0].item()), int(rc[1].item())


def label_trav_components(fmap_np: np.ndarray) -> np.ndarray:
    """8-connected connected components on free cells (fmap > 0.5). Label 0 = blocked."""
    h, w = fmap_np.shape
    free = (fmap_np > 0.5).astype(np.uint8)
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    for r in range(h):
        for c in range(w):
            if free[r, c] == 0 or labels[r, c] != 0:
                continue
            current += 1
            dq: deque[tuple[int, int]] = deque()
            dq.append((r, c))
            labels[r, c] = current
            while dq:
                rr, cc = dq.popleft()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            dq.append((nr, nc))
    return labels


def bridge_disconnected_goals(
    env: Any,
    robot: Any,
    floor: int,
    spawn_xy: tuple[float, float],
) -> list[str]:
    """
    Paint clipped traversable rectangles along spawn -> goal when either:
    - flood-fill says spawn and goal are in different components, or
    - A* from spawn to the preset goal is None (planner/graph mismatch; same label can still be unreachable).
    """
    trav = env.scene.trav_map
    rs, cs = _world_to_rc(trav, spawn_xy[0], spawn_xy[1])
    bridged: list[str] = []
    free_mask = snapshot_wall_free_mask(env, floor=floor)
    map_sz = int(trav.floor_map[floor].shape[0])
    for gx, gy, room in PRESET_GOALS:
        fmap_np = trav.floor_map[floor].detach().cpu().numpy().astype(np.float32)
        labels = label_trav_components(fmap_np)
        if not (0 <= rs < labels.shape[0] and 0 <= cs < labels.shape[1]):
            og.log.warning("bridge_disconnected_goals: spawn cell out of map bounds")
            break
        spawn_comp = int(labels[rs, cs])
        rg, cg = _world_to_rc(trav, gx, gy)
        if not (0 <= rg < labels.shape[0] and 0 <= cg < labels.shape[1]):
            continue
        goal_comp = int(labels[rg, cg])
        path_s, _ = compute_global_path(env, robot, floor, spawn_xy, (gx, gy))
        path_ok = path_s is not None
        same_comp = goal_comp == spawn_comp
        if path_ok and same_comp:
            continue
        for t in np.linspace(0.0, 1.0, 25)[1:-1]:
            wx = float(spawn_xy[0] + t * (gx - spawn_xy[0]))
            wy = float(spawn_xy[1] + t * (gy - spawn_xy[1]))
            xy = th.tensor([wx, wy], dtype=th.float32)
            rc = trav.world_to_map(xy)
            r, c = int(rc[0].item()), int(rc[1].item())
            r0 = max(0, r - DOORWAY_RADIUS_CELLS)
            r1 = min(map_sz - 1, r + DOORWAY_RADIUS_CELLS)
            c0 = max(0, c - DOORWAY_RADIUS_CELLS)
            c1 = min(map_sz - 1, c + DOORWAY_RADIUS_CELLS)
            _paint_rect_clipped(env, floor, r0, r1, c0, c1, free_mask)
        og.log.info(
            f"bridge_disconnected_goals: painted line bridges toward {room} "
            f"(spawn_comp={spawn_comp}, goal_comp={goal_comp}, path_ok={path_ok})"
        )
        bridged.append(room)
    return bridged


def compute_trav_connectivity_meta(
    env: Any,
    robot: Any,
    floor: int,
    spawn_xy: tuple[float, float],
) -> list[dict[str, Any]]:
    """Per PRESET_GOAL: trav component id vs spawn and A* reachability from spawn / chain prev."""
    trav = env.scene.trav_map
    fmap_np = trav.floor_map[floor].detach().cpu().numpy().astype(np.float32)
    labels = label_trav_components(fmap_np)
    rs, cs = _world_to_rc(trav, spawn_xy[0], spawn_xy[1])
    if not (0 <= rs < labels.shape[0] and 0 <= cs < labels.shape[1]):
        return []
    spawn_comp = int(labels[rs, cs])
    entries: list[dict[str, Any]] = []
    prev_xy = spawn_xy
    for gx, gy, room in PRESET_GOALS:
        rg, cg = _world_to_rc(trav, gx, gy)
        if not (0 <= rg < labels.shape[0] and 0 <= cg < labels.shape[1]):
            comp_id = -1
        else:
            comp_id = int(labels[rg, cg])
        path_s, _ = compute_global_path(env, robot, floor, spawn_xy, (gx, gy))
        path_p, _ = compute_global_path(env, robot, floor, prev_xy, (gx, gy))
        entries.append(
            {
                "room": room,
                "component_id": comp_id,
                "spawn_component": spawn_comp,
                "same_component_as_spawn": comp_id == spawn_comp,
                "snap_ok_from_spawn": path_s is not None,
                "snap_ok_from_prev": path_p is not None,
            }
        )
        prev_xy = (gx, gy)
    return entries


def collision_with_world(env, robot) -> bool:
    floors = list(env.scene.object_registry("category", "floors", []))
    ignore_objs = tuple(floors + [robot])
    bodies = robot.states[ContactBodies].get_value(ignore_objs=ignore_objs)
    return len(bodies) > 0


def main():
    parser = argparse.ArgumentParser(
        description="180s autonomous nav demo — Benevolence_2_int (ordered x1..x7 tour; same stack as autonomous_nav_benevolence1.py)."
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Run ~10 s (300 steps) for quick smoke test.",
    )
    parser.add_argument(
        "--no-teleop-camera",
        action="store_true",
        help="Do not enable viewer camera teleoperation (default: enabled; use WASD/mouse to frame the scene).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record first-person POV video from the robot's RGB camera.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run PRESET_GOALS exactly once; stop the sim after the final goal is reached.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "output" / "autonomous_nav_benevolence2_fpv.mp4",
        help="Output path for the recorded FPV video (default: output/autonomous_nav_benevolence2_fpv.mp4).",
    )
    args = parser.parse_args()

    n_steps = 300 if args.short else N_STEPS
    sim_seconds = n_steps / float(ACTION_FREQ)

    og.log.info(
        f"Autonomous navigation: {n_steps} steps ~= {sim_seconds:.1f} s sim time at {ACTION_FREQ} Hz"
    )

    config = build_env_config()
    env = og.Environment(configs=config)

    if not args.no_teleop_camera:
        og.sim.enable_viewer_camera_teleoperation()

    obs, _reset_info = env.reset()
    open_all_doors(env)
    for _ in range(5):
        og.sim.step()
    # Auto-derive open-archway carving points from build_nav_paths.py output so that closet_0<->corridor_0
    # and similar via_doors=[] passages get painted into the runtime trav map.
    auto_passages = load_passage_waypoints_from_nav_paths(NAV_PATHS_JSON_PATH)
    if auto_passages:
        before = len(PASSAGE_WAYPOINTS)
        PASSAGE_WAYPOINTS.extend(auto_passages)
        print(
            f"[PASSAGES] Loaded {len(auto_passages)} archway carve points from {NAV_PATHS_JSON_PATH.name} "
            f"(total now {before + len(auto_passages)})."
        )
    prepare_trav_map_doorways(env, floor=0)
    robot = env.robots[0]
    robot_name = robot.name
    floor = 0
    floor_z = trav_floor_z(env, floor)

    # DummyTask does not place/land the robot — spawn in same trav component as preset goals.
    spawn_near_goals(env, robot, floor, floor_z)
    stabilize_after_land(env, robot_name)
    obs, _ = env.get_obs()

    pos, _ = robot.get_position_orientation()
    print(f"[ROBOT] at ({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f})")
    if float(pos[2]) < floor_z - 1.0:
        print("[WARN] Robot below floor; re-landing.")
        spawn_near_goals(env, robot, floor, floor_z)
        stabilize_after_land(env, robot_name)
        obs, _ = env.get_obs()
        pos, _ = robot.get_position_orientation()
        print(f"[ROBOT] re-landed at ({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f})")

    x, y, yaw = robot_xy_yaw(robot)
    bridged_goal_names = bridge_disconnected_goals(env, robot, floor, (x, y))
    # Bridge paints clipped traversable corridors; if those overlap the sofa NE pinch, re-block it so
    # A* cannot route through that slot after bridging (in-place write; idempotent).
    paint_extra_trav_blockers(env, floor=floor)
    connectivity_entries = compute_trav_connectivity_meta(env, robot, floor, (x, y))
    active_goals, waypoints_snapped_meta = build_snapped_goal_tour(env, robot, floor, (x, y))
    once_ignore_early_exit = bool(args.once and len(active_goals) < 2)
    if once_ignore_early_exit:
        print(
            f"[WARN] Trivial tour: only {len(active_goals)} snapped goal(s); "
            f"--once will NOT early-exit — recording full {n_steps} steps. "
            f"spawn=({x:.2f},{y:.2f}) bridged={bridged_goal_names!r}"
        )
    if not active_goals:
        og.log.warning("No snapped preset goals; falling back to validate_preset_goals.")
        active_goals = validate_preset_goals(env, robot, floor, (x, y))
        waypoints_snapped_meta = []
    if not active_goals:
        og.log.warning("No preset goals reachable from spawn; using raw PRESET_GOALS list.")
        active_goals = list(PRESET_GOALS)
        waypoints_snapped_meta = []

    # If spawn is already within GOAL_REACH_M of the first tour goal (e.g. X1), treat it as pre-reached
    # so the edge-trigger can fire for subsequent goals (avoids spin-at-start).
    pre_reached_goal: tuple[float, float, str] | None = None
    if active_goals and distance_xy(x, y, active_goals[0][0], active_goals[0][1]) < GOAL_REACH_M:
        gx, gy, room = active_goals.pop(0)
        pre_reached_goal = (gx, gy, room)
        for m in waypoints_snapped_meta:
            if m.get("room") == room and m.get("status") == "ok":
                m["status"] = "pre_reached"
                break
        print(
            f"[SPAWN_AT_X1] treating {room} as already reached; next = "
            f"{active_goals[0][2] if active_goals else 'none'}"
        )
        once_ignore_early_exit = bool(args.once and len(active_goals) < 2)
        if once_ignore_early_exit and pre_reached_goal:
            print(
                f"[WARN] Trivial tour after spawn pre-reach: only {len(active_goals)} goal(s); "
                f"--once will NOT early-exit — recording full {n_steps} steps."
            )
    if not active_goals:
        og.log.warning("No goals left after spawn pre-reach; restoring PRESET_GOALS.")
        active_goals = list(PRESET_GOALS)
        waypoints_snapped_meta = []

    nav_checks_meta: list[dict[str, Any]] = []
    if active_goals:
        nav_checks_meta = run_nav_checks(env, robot, floor, (x, y), active_goals)

    goal_idx = 0
    goal_x, goal_y, goal_room, path, goal_idx, geo_init = first_reachable_from_index(
        env, robot, floor, (x, y), active_goals, goal_idx
    )
    env.task.set_start_marker(x, y, floor_z)
    env.task.set_goal_marker(goal_x, goal_y, floor_z)
    print(f"[GOAL] room={goal_room!r} target=({goal_x:.2f},{goal_y:.2f})")
    last_geodesic_m: float | None = geo_init

    diag_dir = args.output.parent
    diag_dir.mkdir(parents=True, exist_ok=True)
    events_path = diag_dir / EVENTS_LOG_NAME
    nav_csv_path = diag_dir / NAV_CSV_NAME
    summary_path = diag_dir / SUMMARY_JSON_NAME
    events_fp = open(events_path, "w", encoding="utf-8")
    diag = RunDiagnostics(events_fp=events_fp)
    diag.log_event("waypoints_tour", entries=waypoints_snapped_meta)
    diag.log_event("trav_connectivity", entries=connectivity_entries)
    diag.log_event("nav_checks", entries=nav_checks_meta)
    if pre_reached_goal is not None:
        diag.log_event(
            "goal_pre_reached",
            goal_room=pre_reached_goal[2],
            x=x,
            y=y,
            sim_t=0.0,
        )

    sx, sy = float(pos[0].item()), float(pos[1].item())
    spawn_i = min(
        range(len(SPAWN_CANDIDATES)),
        key=lambda i: distance_xy(sx, sy, SPAWN_CANDIDATES[i][0], SPAWN_CANDIDATES[i][1]),
    )
    diag.log_event(
        "spawn_chosen",
        x=sx,
        y=sy,
        candidate_index=spawn_i,
        candidate=list(SPAWN_CANDIDATES[spawn_i]),
        active_goals=[{"room": r, "xy": [gx, gy]} for gx, gy, r in active_goals],
    )
    diag.log_event(
        "goal_selected",
        goal_idx=goal_idx,
        goal_room=goal_room,
        goal_x=goal_x,
        goal_y=goal_y,
        has_astar_path=path is not None,
        geodesic_m=last_geodesic_m,
    )

    csv_fp = open(nav_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fp)
    csv_writer.writerow(
        [
            "step",
            "sim_t",
            "x",
            "y",
            "yaw",
            "goal_idx",
            "goal_room",
            "goal_x",
            "goal_y",
            "dist_to_goal",
            "path_len_pts",
            "path_dev_m",
            "geodesic_m",
            "look_ahead_eff",
            "path_progress_m",
            "dist_goal_delta_m",
            "v_cmd",
            "w_cmd",
            "cmd_src",
            "d_front",
            "d_left",
            "d_right",
            "d_min360",
            "contact",
            "stuck_skips_this_goal",
            "replan_count_total",
            "astar_fail_count_total",
        ]
    )

    replan_counter = 0
    goals_reached = 1 if pre_reached_goal is not None else 0
    goals_skipped = 0
    collision_steps = 0
    total_path_m = 0.0
    prev_x, prev_y = x, y
    pos_hist: deque[tuple[float, float]] = deque(maxlen=STUCK_WINDOW_STEPS)
    dist_goal_hist: deque[float] = deque(maxlen=STUCK_WINDOW_STEPS)
    contact_stall_steps = 0
    stuck_skips_this_goal = 0
    prev_contact = False
    vfh_active = False
    prev_strong_vfh = False
    prev_w_smoothed = 0.0
    recovery_phase: str | None = None
    recovery_steps_left = 0
    recovery_rotate_steps = 0
    recovery_target_yaw = 0.0
    goal_segment_start_t = 0.0
    goal_segment_path_m = 0.0
    goal_segment_collisions = 0
    per_goal_stats: list[dict[str, Any]] = []
    if pre_reached_goal is not None:
        per_goal_stats.append(
            {
                "goal_idx": 0,
                "room": pre_reached_goal[2],
                "x": pre_reached_goal[0],
                "y": pre_reached_goal[1],
                "status": "pre_reached",
                "time_s": 0.0,
                "path_m": 0.0,
                "collision_steps": 0,
            }
        )
    # Edge-trigger goal reach: avoid incrementing goals_reached every step when already inside GOAL_REACH_M
    # (e.g. single goal at spawn after other presets failed to snap).
    prev_dist_to_goal = float("inf")
    _sx, _sy, _ = robot_xy_yaw(robot)
    dist_spawn_to_goal = distance_xy(_sx, _sy, goal_x, goal_y)
    if dist_spawn_to_goal < GOAL_REACH_M:
        prev_dist_to_goal = dist_spawn_to_goal

    writer: cv2.VideoWriter | None = None
    video_path: Path = args.output
    if args.record:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[RECORD] FPV video -> {video_path}")

    print("=" * 72)
    print("TIMER (sim time): show this window during your 180s recording.")
    print("Commands: same stack as Part A; this script runs the autonomous segment.")
    if not args.no_teleop_camera:
        print("Viewer: keyboard camera teleoperation enabled (use --no-teleop-camera to disable).")
    print("=" * 72)

    step = -1
    terminated = False
    truncated = False
    try:
        for step in range(n_steps):
            sim_t = (step + 1) / float(ACTION_FREQ)
            x0, y0, yaw0 = robot_xy_yaw(robot)
            cmd_src = "pp"
            v_cmd, w_cmd = 0.0, 0.0
            geodesic_m = last_geodesic_m
            look_ahead_eff = LOOK_AHEAD_M

            if step % (5 * ACTION_FREQ) == 0:
                print(
                    f"[TIMER] sim_t={step / float(ACTION_FREQ):6.2f}s / {sim_seconds:.0f}s | "
                    f"goals_reached={goals_reached} | goal=({goal_x:.2f},{goal_y:.2f})"
                )

            if args.record:
                rgb = extract_rgb_from_obs(obs[robot_name])
                if rgb is not None:
                    bgr = rgb_to_bgr_uint8(rgb)
                    if writer is None:
                        hh, ww = bgr.shape[0], bgr.shape[1]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(video_path), fourcc, float(ACTION_FREQ), (ww, hh))
                    writer.write(bgr)

            if recovery_phase == "backup":
                v_cmd = RECOVERY_BACKUP_V
                w_cmd = 0.0
                cmd_src = "recovery"
                recovery_steps_left -= 1
                if recovery_steps_left <= 0:
                    xb, yb, yawb = robot_xy_yaw(robot)
                    scan_b = extract_scan_from_obs(obs[robot_name])
                    if scan_b is not None:
                        dist_b = normalized_scan_to_meters(scan_b)
                        des_b = bearing_goal_desired_heading_rel(xb, yb, yawb, goal_x, goal_y)
                        meta_b = vfh_recovery_heading_meta(dist_b, des_b)
                        delta = float(meta_b.get("chosen_bin_rad", des_b))
                    else:
                        delta = bearing_goal_desired_heading_rel(xb, yb, yawb, goal_x, goal_y)
                    recovery_target_yaw = wrap_angle_pi(yawb + delta)
                    recovery_phase = "rotate"
                    recovery_rotate_steps = RECOVERY_ROTATE_MAX_STEPS
                    diag.log_event("recovery_end", phase="backup", duration_steps=RECOVERY_BACKUP_STEPS)
                    diag.log_event(
                        "recovery_begin",
                        phase="rotate",
                        max_steps=RECOVERY_ROTATE_MAX_STEPS,
                        target_yaw_deg=math.degrees(recovery_target_yaw),
                    )

            elif recovery_phase == "rotate":
                err_r = wrap_angle_pi(recovery_target_yaw - yaw0)
                recovery_rotate_steps -= 1
                if abs(err_r) < RECOVERY_ALIGN_TOL_RAD or recovery_rotate_steps < 0:
                    recovery_phase = None
                    xr, yr, _yr = robot_xy_yaw(robot)
                    path, geo_r = compute_global_path(env, robot, floor, (xr, yr), (goal_x, goal_y))
                    diag.replan_count += 1
                    replan_counter = 0
                    if path is None:
                        diag.astar_fail_count += 1
                        diag.log_event(
                            "astar_fail",
                            from_xy=[xr, yr],
                            to_xy=[goal_x, goal_y],
                            reason="path_is_none",
                        )
                    else:
                        last_geodesic_m = (
                            float(geo_r.item()) if isinstance(geo_r, th.Tensor) else float(geo_r)
                        )
                        geodesic_m = last_geodesic_m
                    pos_hist.clear()
                    dist_goal_hist.clear()
                    diag.log_event(
                        "recovery_end",
                        phase="rotate",
                        aligned=abs(err_r) < RECOVERY_ALIGN_TOL_RAD,
                        steps_remaining=recovery_rotate_steps,
                    )
                    v_cmd, w_cmd = 0.0, 0.0
                    cmd_src = "recovery"
                else:
                    w_cmd = max(-1.0, min(1.0, KP_ANG * err_r / math.pi))
                    v_cmd = 0.0
                    cmd_src = "recovery"

            else:
                replan_counter += 1
                dev = path_deviation_m(path, x0, y0) if path is not None else PATH_DEVIATION_REPLAN_M + 1.0
                if path is None or replan_counter >= REPLAN_EVERY_STEPS or dev > PATH_DEVIATION_REPLAN_M:
                    path, geo_p = compute_global_path(env, robot, floor, (x0, y0), (goal_x, goal_y))
                    diag.replan_count += 1
                    replan_counter = 0
                    if path is None:
                        diag.astar_fail_count += 1
                        diag.log_event(
                            "astar_fail",
                            from_xy=[x0, y0],
                            to_xy=[goal_x, goal_y],
                            reason="path_is_none",
                        )
                    else:
                        last_geodesic_m = (
                            float(geo_p.item()) if isinstance(geo_p, th.Tensor) else float(geo_p)
                        )
                        geodesic_m = last_geodesic_m

                scan_norm = extract_scan_from_obs(obs[robot_name])
                d_lidar_front: float | None = None
                d_front = d_left = d_right = d_min360 = float(LIDAR_MAX_RANGE)
                vfh_blend: VfhBlendCommand | None = None
                vfh_meta: dict[str, Any] = {}
                if scan_norm is not None:
                    dist_m = normalized_scan_to_meters(scan_norm)
                    d_front, d_left, d_right, d_min360 = lidar_sector_mins(dist_m)
                    d_lidar_front = d_front
                    diag.record_d_front(d_front)
                    look_ahead_eff = effective_lookahead_m(d_lidar_front)
                    if vfh_active:
                        if d_front > SLOW_DIST_EXIT:
                            vfh_active = False
                            diag.log_event("vfh_active_exited", d_front=d_front)
                    else:
                        if d_front < SLOW_DIST_ENTER:
                            vfh_active = True
                            diag.log_event("vfh_active_entered", d_front=d_front)

                    if path is not None:
                        des = pure_pursuit_desired_heading_rel(x0, y0, yaw0, path, d_lidar_front)
                    else:
                        des = bearing_goal_desired_heading_rel(x0, y0, yaw0, goal_x, goal_y)
                    if des is None:
                        des = 0.0
                    if vfh_active:
                        vfh_blend, vfh_meta = vfh_goal_biased_when_active(dist_m, des, d_front)
                else:
                    if path is not None:
                        des = pure_pursuit_desired_heading_rel(x0, y0, yaw0, path, None)
                    else:
                        des = bearing_goal_desired_heading_rel(x0, y0, yaw0, goal_x, goal_y)
                    if des is None:
                        des = 0.0

                if path is not None:
                    v_pp, w_pp = pure_pursuit_cmd(x0, y0, yaw0, path, d_lidar_front)
                    cmd_src = "pp"
                else:
                    v_pp, w_pp = bearing_to_goal_cmd(x0, y0, yaw0, goal_x, goal_y)
                    cmd_src = "bearing"

                dist_pre_goal = distance_xy(x0, y0, goal_x, goal_y)
                if vfh_active and vfh_blend is not None:
                    bypass_pp_slow = (
                        vfh_blend.mode == "pp_slow"
                        and path is not None
                        and dist_pre_goal < TRUST_ASTAR_NEAR_GOAL_M
                    )
                    if bypass_pp_slow:
                        v_cmd, w_cmd = v_pp, w_pp
                        cmd_src = "pp"
                        prev_strong_vfh = False
                    else:
                        v_cmd, w_cmd = apply_vfh_blend(v_pp, w_pp, vfh_blend)
                        if vfh_blend.mode == "pp_slow":
                            diag.pp_slow_steps += 1
                            cmd_src = "pp_slow"
                            prev_strong_vfh = False
                        elif vfh_blend.mode == "vfh":
                            diag.vfh_override_steps += 1
                            cmd_src = "vfh"
                            if not prev_strong_vfh:
                                diag.log_event(
                                    "vfh_override",
                                    desired_heading_deg=vfh_meta.get("desired_heading_deg"),
                                    chosen_bin_deg=vfh_meta.get("chosen_bin_deg"),
                                    chosen_clearance_m=vfh_meta.get("chosen_clearance_m"),
                                    emergency=False,
                                    d_front=vfh_meta.get("d_front"),
                                    d_beam_pp=vfh_meta.get("d_beam_pp"),
                                    mode=vfh_meta.get("mode"),
                                )
                            prev_strong_vfh = True
                        else:
                            diag.vfh_override_steps += 1
                            cmd_src = "emergency"
                            if not prev_strong_vfh:
                                diag.log_event(
                                    "vfh_override",
                                    desired_heading_deg=vfh_meta.get("desired_heading_deg"),
                                    chosen_bin_deg=vfh_meta.get("chosen_bin_deg"),
                                    chosen_clearance_m=vfh_meta.get("chosen_clearance_m"),
                                    emergency=True,
                                    d_front=vfh_meta.get("d_front"),
                                    d_beam_pp=vfh_meta.get("d_beam_pp"),
                                    mode=vfh_meta.get("mode"),
                                )
                            prev_strong_vfh = True
                else:
                    v_cmd, w_cmd = v_pp, w_pp
                    prev_strong_vfh = False

                # Single-step backup to break wall-hug (VFH can output tiny v while in contact).
                if (
                    recovery_phase is None
                    and prev_contact
                    and vfh_active
                    and vfh_blend is not None
                    and vfh_blend.mode in ("pp_slow", "vfh")
                    and v_cmd < 0.1
                ):
                    v_cmd = RECOVERY_BACKUP_V
                    prev_w_smoothed = 0.0

            # Smooth only during reactive avoidance; crisp PP/bearing avoids living-room stalls.
            if recovery_phase is None:
                if cmd_src in ("pp_slow", "vfh", "emergency"):
                    w_cmd = W_CMD_EMA_NEW * w_cmd + (1.0 - W_CMD_EMA_NEW) * prev_w_smoothed
                    prev_w_smoothed = w_cmd
                    if abs(w_cmd) < W_CMD_DEADBAND:
                        w_cmd = 0.0
                        prev_w_smoothed = 0.0
                else:
                    prev_w_smoothed = w_cmd
            else:
                prev_w_smoothed = w_cmd

            action = {robot_name: th.tensor([v_cmd, w_cmd], dtype=th.float32)}
            obs, _reward, terminated, truncated, _info = env.step(action)

            x, y, yaw = robot_xy_yaw(robot)
            dist_goal = distance_xy(x, y, goal_x, goal_y)
            path_len = int(path.shape[0]) if path is not None and path.numel() > 0 else 0
            dev_m = path_deviation_m(path, x, y) if path is not None else 0.0
            scan_post = extract_scan_from_obs(obs[robot_name])
            if scan_post is not None:
                dm = normalized_scan_to_meters(scan_post)
                d_front, d_left, d_right, d_min360 = lidar_sector_mins(dm)
            else:
                d_front = d_left = d_right = d_min360 = float(LIDAR_MAX_RANGE)

            contact_now = collision_with_world(env, robot)
            path_prog_m = (
                path_arclength_to_closest(path, x, y) if path is not None else float("nan")
            )

            if contact_now and abs(v_cmd) < CONTACT_STALL_V_THRESH and abs(w_cmd) < CONTACT_STALL_W_THRESH:
                contact_stall_steps += 1
            else:
                contact_stall_steps = 0

            if recovery_phase is None and contact_stall_steps >= CONTACT_STALL_TRIGGER_STEPS:
                diag.log_event(
                    "stuck_detected",
                    x=x,
                    y=y,
                    goal_idx=goal_idx,
                    mode="contact_stall",
                    contact_stall_steps=contact_stall_steps,
                )
                diag.contact_stall_triggers += 1
                recovery_phase = "backup"
                recovery_steps_left = RECOVERY_BACKUP_STEPS
                vfh_active = False
                prev_strong_vfh = False
                prev_w_smoothed = 0.0
                contact_stall_steps = 0
                pos_hist.clear()
                dist_goal_hist.clear()
                diag.recovery_episodes += 1
                diag.log_event("recovery_begin", phase="backup", duration_steps=RECOVERY_BACKUP_STEPS)

            seg_d = distance_xy(prev_x, prev_y, x, y)
            total_path_m += seg_d
            goal_segment_path_m += seg_d
            prev_x, prev_y = x, y

            if contact_now:
                collision_steps += 1
                goal_segment_collisions += 1
            if contact_now and not prev_contact:
                diag.log_event("contact_enter", x=x, y=y, sim_t=sim_t)
            if not contact_now and prev_contact:
                diag.log_event("contact_exit", x=x, y=y, sim_t=sim_t)
            prev_contact = contact_now

            pos_hist.append((x, y))
            dist_goal_hist.append(dist_goal)

            dist_goal_delta_str = ""
            if len(dist_goal_hist) >= STUCK_WINDOW_STEPS:
                dist_goal_delta_str = f"{dist_goal_hist[0] - dist_goal_hist[-1]:.4f}"

            csv_writer.writerow(
                [
                    step,
                    f"{sim_t:.4f}",
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{yaw:.4f}",
                    goal_idx,
                    goal_room,
                    f"{goal_x:.4f}",
                    f"{goal_y:.4f}",
                    f"{dist_goal:.4f}",
                    path_len,
                    f"{dev_m:.4f}",
                    f"{geodesic_m if geodesic_m is not None else ''}",
                    f"{look_ahead_eff:.4f}",
                    f"{path_prog_m:.4f}" if path is not None else "",
                    dist_goal_delta_str,
                    f"{v_cmd:.4f}",
                    f"{w_cmd:.4f}",
                    cmd_src,
                    f"{d_front:.4f}",
                    f"{d_left:.4f}",
                    f"{d_right:.4f}",
                    f"{d_min360:.4f}",
                    int(contact_now),
                    stuck_skips_this_goal,
                    diag.replan_count,
                    diag.astar_fail_count,
                ]
            )
            if step % CSV_FLUSH_EVERY == 0:
                csv_fp.flush()

            if prev_dist_to_goal >= GOAL_REACH_M and dist_goal < GOAL_REACH_M:
                goals_reached += 1
                print(f"*** Reached goal #{goals_reached} at sim_t={sim_t:.2f}s — next preset goal ***")
                per_goal_stats.append(
                    {
                        "goal_idx": goal_idx,
                        "room": goal_room,
                        "x": goal_x,
                        "y": goal_y,
                        "status": "reached",
                        "time_s": sim_t - goal_segment_start_t,
                        "path_m": goal_segment_path_m,
                        "collision_steps": goal_segment_collisions,
                    }
                )
                diag.log_event("goal_reached", goal_idx=goal_idx, goal_room=goal_room, sim_t=sim_t)
                if (
                    args.once
                    and len(active_goals) > 0
                    and goal_idx == len(active_goals) - 1
                    and not once_ignore_early_exit
                ):
                    print(f"*** Final goal reached at sim_t={sim_t:.2f}s; stopping (--once) ***")
                    break
                goal_idx = (goal_idx + 1) % len(active_goals)
                goal_x, goal_y, goal_room, path, goal_idx, geo_new = first_reachable_from_index(
                    env, robot, floor, (x, y), active_goals, goal_idx
                )
                last_geodesic_m = geo_new
                replan_counter = 0
                stuck_skips_this_goal = 0
                pos_hist.clear()
                dist_goal_hist.clear()
                contact_stall_steps = 0
                recovery_phase = None
                vfh_active = False
                prev_strong_vfh = False
                prev_w_smoothed = 0.0
                goal_segment_start_t = sim_t
                goal_segment_path_m = 0.0
                goal_segment_collisions = 0
                env.task.set_start_marker(x, y, floor_z)
                env.task.set_goal_marker(goal_x, goal_y, floor_z)
                diag.log_event(
                    "goal_selected",
                    goal_idx=goal_idx,
                    goal_room=goal_room,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    has_astar_path=path is not None,
                    geodesic_m=last_geodesic_m,
                )
                print(f"[GOAL] room={goal_room!r} target=({goal_x:.2f},{goal_y:.2f})")

            elif recovery_phase is None and len(dist_goal_hist) >= STUCK_WINDOW_STEPS:
                euclid_win = distance_xy(pos_hist[0][0], pos_hist[0][1], x, y)
                d0 = dist_goal_hist[0]
                d1 = dist_goal_hist[-1]
                dist_goal_delta = d0 - d1
                stuck = dist_goal_delta < STUCK_PATH_PROGRESS_THRESHOLD_M
                stuck_mode = "path_dist_to_goal"
                if stuck:
                    diag.log_event(
                        "stuck_detected",
                        x=x,
                        y=y,
                        goal_idx=goal_idx,
                        stuck_skips_before=stuck_skips_this_goal,
                        mode=stuck_mode,
                        euclid_m=euclid_win,
                        dist_goal_delta_m=dist_goal_delta,
                    )
                    stuck_skips_this_goal += 1
                    if stuck_skips_this_goal >= MAX_STUCK_SKIPS_PER_GOAL:
                        goals_skipped += 1
                        print(
                            f"[STUCK] skipping goal {goal_room!r} ({goal_x:.2f},{goal_y:.2f}) "
                            f"after {stuck_skips_this_goal} stuck episodes at sim_t={sim_t:.2f}s"
                        )
                        per_goal_stats.append(
                            {
                                "goal_idx": goal_idx,
                                "room": goal_room,
                                "x": goal_x,
                                "y": goal_y,
                                "status": "skipped",
                                "reason": "stuck_max",
                                "time_s": sim_t - goal_segment_start_t,
                                "path_m": goal_segment_path_m,
                                "collision_steps": goal_segment_collisions,
                            }
                        )
                        diag.log_event(
                            "goal_skipped",
                            goal_idx=goal_idx,
                            goal_room=goal_room,
                            reason="stuck_max",
                        )
                        goal_idx = (goal_idx + 1) % len(active_goals)
                        goal_x, goal_y, goal_room, path, goal_idx, geo_new = first_reachable_from_index(
                            env, robot, floor, (x, y), active_goals, goal_idx
                        )
                        last_geodesic_m = geo_new
                        replan_counter = 0
                        stuck_skips_this_goal = 0
                        pos_hist.clear()
                        dist_goal_hist.clear()
                        contact_stall_steps = 0
                        recovery_phase = None
                        vfh_active = False
                        prev_strong_vfh = False
                        prev_w_smoothed = 0.0
                        goal_segment_start_t = sim_t
                        goal_segment_path_m = 0.0
                        goal_segment_collisions = 0
                        env.task.set_start_marker(x, y, floor_z)
                        env.task.set_goal_marker(goal_x, goal_y, floor_z)
                        diag.log_event(
                            "goal_selected",
                            goal_idx=goal_idx,
                            goal_room=goal_room,
                            goal_x=goal_x,
                            goal_y=goal_y,
                            has_astar_path=path is not None,
                            geodesic_m=last_geodesic_m,
                        )
                        print(f"[GOAL] room={goal_room!r} target=({goal_x:.2f},{goal_y:.2f})")
                    else:
                        print(
                            f"[STUCK] detected near ({x:.2f},{y:.2f}), goal={goal_room!r} "
                            f"attempt {stuck_skips_this_goal}/{MAX_STUCK_SKIPS_PER_GOAL}, recovery"
                        )
                        recovery_phase = "backup"
                        recovery_steps_left = RECOVERY_BACKUP_STEPS
                        vfh_active = False
                        prev_strong_vfh = False
                        prev_w_smoothed = 0.0
                        diag.recovery_episodes += 1
                        diag.log_event("recovery_begin", phase="backup", duration_steps=RECOVERY_BACKUP_STEPS)
                    pos_hist.clear()
                    dist_goal_hist.clear()
                    contact_stall_steps = 0

            prev_dist_to_goal = dist_goal

            if terminated or truncated:
                og.log.warning("Episode ended unexpectedly (should not happen with DummyTask).")
                break

        diag.log_event(
            "run_complete",
            steps=step + 1 if step >= 0 else 0,
            goals_reached=goals_reached,
            goals_skipped=goals_skipped,
        )
    finally:
        events_fp.close()
        csv_fp.close()

    effective_sim_seconds = (step + 1) / float(ACTION_FREQ) if step >= 0 else 0.0

    samples = sorted(diag.d_front_samples)
    n_s = len(samples)
    p05 = samples[int(0.05 * (n_s - 1))] if n_s > 0 else None
    d_mean = sum(samples) / n_s if n_s > 0 else None
    d_min = min(samples) if n_s > 0 else None

    def _rel_repo(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(_REPO_ROOT.resolve()))
        except ValueError:
            return str(p)

    summary: dict[str, Any] = {
        "sim_seconds": effective_sim_seconds,
        "goals_reached": goals_reached,
        "goals_skipped": goals_skipped,
        "collision_steps": collision_steps,
        "total_path_m": total_path_m,
        "replan_count": diag.replan_count,
        "astar_fail_count": diag.astar_fail_count,
        "vfh_override_steps": diag.vfh_override_steps,
        "pp_slow_steps": diag.pp_slow_steps,
        "recovery_episodes": diag.recovery_episodes,
        "contact_stall_triggers": diag.contact_stall_triggers,
        "waypoints_snapped": waypoints_snapped_meta,
        "nav_checks": nav_checks_meta,
        "per_goal": per_goal_stats,
        "lidar": {
            "d_front_mean": d_mean,
            "d_front_min": d_min,
            "d_front_p05": p05,
            "d_front_num_samples": n_s,
        },
        "artifacts": {
            "video_mp4": _rel_repo(video_path),
            "nav_csv": _rel_repo(nav_csv_path),
            "events": _rel_repo(events_path),
            "summary_json": _rel_repo(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if writer is not None:
        writer.release()
        print(f"[RECORD] wrote {video_path} ({step + 1} frames)")
        finalize_mp4(video_path)

    print("=" * 72)
    print("SUMMARY")
    print(f"  Sim time covered: {effective_sim_seconds:.1f} s")
    print(f"  Goals reached: {goals_reached}")
    print(f"  Goals skipped (stuck): {goals_skipped}")
    print(f"  Approx. path length: {total_path_m:.2f} m")
    print(f"  Steps with contact (non-floor): {collision_steps}")
    print(f"  Replans: {diag.replan_count} | A* failures: {diag.astar_fail_count}")
    print(
        f"  VFH steering steps: {diag.vfh_override_steps} | PP-slow steps: {diag.pp_slow_steps} "
        f"| Recovery episodes: {diag.recovery_episodes} | Contact-stall triggers: {diag.contact_stall_triggers}"
    )
    print(f"  Diagnostics: {nav_csv_path.name}, {EVENTS_LOG_NAME}, {SUMMARY_JSON_NAME}")
    print(json.dumps(summary, indent=2))
    print("=" * 72)

    og.shutdown()


if __name__ == "__main__":
    main()
