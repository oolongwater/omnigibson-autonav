#!/usr/bin/env python3
"""
Autonomous navigation for 180 continuous simulation seconds in OmniGibson.

Uses A* on the traversable map (with-objects trav + painted doorways) + pure pursuit on the path,
bearing-to-goal when A* fails, and 2D LiDAR avoidance. Walls are solid (collision on); door panels
are opened then non-collidable so the opening in the wall mesh is passable.

Run from anywhere; ensures BEHAVIOR-1K/OmniGibson is on sys.path.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
REPLAN_EVERY_STEPS = 30
PATH_DEVIATION_REPLAN_M = 1.2
GOAL_REACH_M = 0.55
KP_ANG = 2.2
V_MAX_NORM = 0.85
W_MAX_NORM = 0.9

# Obstacle avoidance
FRONT_HALF_DEG = 35.0
SIDE_DEG = 55.0
SLOW_DIST = 0.40
STOP_DIST = 0.15

# Stabilization after landing
POST_LAND_SIM_STEPS = 15
POST_LAND_RENDER_PASSES = 3

# Goal marker appearance (matches PointNavigationTask defaults)
MARKER_RADIUS = 0.36
MARKER_HEIGHT = 0.01

# Preset waypoints (world x, y, room label) — validated with A* from robot pose at runtime.
PRESET_GOALS: list[tuple[float, float, str]] = [
    (-0.70, 2.30, "kitchen_0"),
    (0.80, -0.80, "living_room_0"),
    (1.30, 3.00, "entryway_0"),
]

# Fallback rectangles (row0, row1, col0, col1) if no door objects are in the scene registry.
DOORWAYS: list[tuple[int, int, int, int]] = [
    (54, 59, 28, 38),  # kitchen <-> living_room
    (54, 59, 48, 58),  # entryway <-> living_room
    (22, 50, 18, 24),  # bedroom <-> living_room
]

# Half-width in trav-map cells around each door world position when painting A* connectivity.
DOORWAY_RADIUS_CELLS = 8

# Spawn candidates: corridor + living room (reachable with walls + open doors + trav doorway paint).
SPAWN_CANDIDATES: list[tuple[float, float]] = [
    (0.0, 2.5),
    (0.5, 2.7),
    (-0.5, 2.3),
    (0.0, 1.0),
    (0.0, 0.0),
    (0.8, -0.8),
]


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
            "scene_model": "Rs_int",
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


def first_reachable_from_index(
    env,
    robot,
    floor: int,
    start_xy: tuple[float, float],
    active_goals: list[tuple[float, float, str]],
    start_index: int,
) -> tuple[float, float, str | None, th.Tensor | None, int]:
    """Try active_goals[start_index], then wrap, until one has a path."""
    n = len(active_goals)
    if n == 0:
        return 0.0, 0.0, None, None, 0
    for k in range(n):
        j = (start_index + k) % n
        gx, gy, room = active_goals[j]
        path, _ = compute_global_path(env, robot, floor, start_xy, (gx, gy))
        if path is not None:
            return gx, gy, room, path, j
    gx, gy, room = active_goals[start_index % n]
    return gx, gy, room, None, start_index % n


def trav_floor_z(env, floor: int = 0) -> float:
    """Floor height from traversability map (Scene.get_floor_height is 0.0 for InteractiveTraversableScene)."""
    h = env.scene.trav_map.floor_heights[floor]
    return float(h.item()) if isinstance(h, th.Tensor) else float(h)


def spawn_near_goals(env, robot, floor: int, floor_z: float):
    """
    Land the robot on a pose that shares a traversable component with at least one PRESET_GOAL
    (avoids random spawn on disconnected islands before doorways are opened in trav map).
    """
    z_off = env.initial_pos_z_offset
    quat = T.euler2quat(th.tensor([0.0, 0.0, 0.0]))
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


def pure_pursuit_cmd(x: float, y: float, yaw: float, path: th.Tensor) -> tuple[float, float]:
    """Returns normalized (lin_vel, ang_vel) in [-1,1]."""
    target = lookahead_point(path, x, y, LOOK_AHEAD_M)
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


def lidar_avoidance_cmd(dist_m: th.Tensor) -> tuple[float, float] | None:
    """
    Reactive avoidance from raw range readings (meters).
    Detects walls and furniture in front / side sectors; no symmetric bypass (solid walls need real reactions).
    """
    n = dist_m.shape[0]
    if n < 8:
        return None
    angles = th.linspace(-math.pi, math.pi, n, device=dist_m.device, dtype=dist_m.dtype)
    deg = angles * (180.0 / math.pi)

    front = (deg.abs() <= FRONT_HALF_DEG) | (deg.abs() >= 360.0 - FRONT_HALF_DEG)
    left = (deg > FRONT_HALF_DEG) & (deg < SIDE_DEG)
    right = (deg < -FRONT_HALF_DEG) & (deg > -SIDE_DEG)

    d_front = float(th.min(dist_m[front]).item()) if bool(front.any()) else LIDAR_MAX_RANGE
    d_left = float(th.min(dist_m[left]).item()) if bool(left.any()) else LIDAR_MAX_RANGE
    d_right = float(th.min(dist_m[right]).item()) if bool(right.any()) else LIDAR_MAX_RANGE

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
    if v_sc == 0.0 and abs(w_av) > 0.5:  # emergency rotate
        return 0.0, max(-1.0, min(1.0, w_av))
    v = max(-1.0, min(1.0, v_pp * v_sc))
    w = max(-1.0, min(1.0, w_pp + w_av))
    return v, w


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
    # Opened door panels still narrow the gap; make doors non-collidable (all links via EntityPrim).
    for door in all_doors:
        try:
            door.visual_only = True
            og.log.info(f"Set visual_only (no collision) on door: {door.name}")
        except Exception as e:
            og.log.warning(f"Could not set visual_only on door {door.name}: {e}")


def paint_doorways_from_doors(env, floor: int = 0) -> int:
    """Paint trav-map rectangles centered on each door's world position. Returns number of door objects painted."""
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
        og.log.info(f"Painted doorway for {door.name} at map[{r0}:{r1},{c0}:{c1}]")
        count += 1
    og.log.info(f"Painted {count} doorway(s) in trav map from door positions.")
    return count


def paint_hardcoded_doorways(env, floor: int = 0) -> None:
    """Paint static DOORWAYS rectangles as traversable (fallback when no door objects in registry)."""
    trav = env.scene.trav_map.floor_map[floor]
    for r0, r1, c0, c1 in DOORWAYS:
        trav[r0 : r1 + 1, c0 : c1 + 1] = 255
    og.log.info(f"Painted {len(DOORWAYS)} hardcoded doorway rectangle(s) in trav map.")


def prepare_trav_map_doorways(env, floor: int = 0) -> None:
    """Paint A* at real door positions, then union hardcoded DOORWAYS (Rs_int room bridges)."""
    n = paint_doorways_from_doors(env, floor=floor)
    if n == 0:
        og.log.warning("No door objects found in registry; trav map uses hardcoded DOORWAYS only.")
    paint_hardcoded_doorways(env, floor=floor)


def collision_with_world(env, robot) -> bool:
    floors = list(env.scene.object_registry("category", "floors", []))
    ignore_objs = tuple(floors + [robot])
    bodies = robot.states[ContactBodies].get_value(ignore_objs=ignore_objs)
    return len(bodies) > 0


def main():
    parser = argparse.ArgumentParser(description="180s autonomous nav demo for OmniGibson take-home.")
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
        "--output",
        type=Path,
        default=_REPO_ROOT / "output" / "autonomous_nav_fpv.mp4",
        help="Output path for the recorded FPV video (default: output/autonomous_nav_fpv.mp4).",
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
    active_goals = validate_preset_goals(env, robot, floor, (x, y))
    if not active_goals:
        og.log.warning("No preset goals reachable from spawn; using raw PRESET_GOALS list.")
        active_goals = list(PRESET_GOALS)

    goal_idx = 0
    goal_x, goal_y, goal_room, path, goal_idx = first_reachable_from_index(
        env, robot, floor, (x, y), active_goals, goal_idx
    )

    env.task.set_start_marker(x, y, floor_z)
    env.task.set_goal_marker(goal_x, goal_y, floor_z)
    print(f"[GOAL] room={goal_room!r} target=({goal_x:.2f},{goal_y:.2f})")

    replan_counter = 0
    goals_reached = 0
    collision_steps = 0
    total_path_m = 0.0
    prev_x, prev_y = x, y

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

    for step in range(n_steps):
        sim_t = (step + 1) / float(ACTION_FREQ)
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

        x, y, yaw = robot_xy_yaw(robot)

        scan_norm = extract_scan_from_obs(obs[robot_name])
        if scan_norm is not None:
            dist_m = normalized_scan_to_meters(scan_norm)
            avoid = lidar_avoidance_cmd(dist_m)
        else:
            avoid = None

        # Re-plan
        replan_counter += 1
        dev = path_deviation_m(path, x, y) if path is not None else PATH_DEVIATION_REPLAN_M + 1.0
        if path is None or replan_counter >= REPLAN_EVERY_STEPS or dev > PATH_DEVIATION_REPLAN_M:
            path, _ = compute_global_path(env, robot, floor, (x, y), (goal_x, goal_y))
            replan_counter = 0
            # If path is None, keep same goal/marker and retry on next replan (no blinking).

        # Follow A* polyline through door openings; bearing only when no graph path.
        if path is not None:
            v_pp, w_pp = pure_pursuit_cmd(x, y, yaw, path)
        else:
            v_pp, w_pp = bearing_to_goal_cmd(x, y, yaw, goal_x, goal_y)
        v_cmd, w_cmd = blend_avoidance(v_pp, w_pp, avoid)

        action = {robot_name: th.tensor([v_cmd, w_cmd], dtype=th.float32)}
        obs, reward, terminated, truncated, info = env.step(action)

        x, y, yaw = robot_xy_yaw(robot)
        total_path_m += distance_xy(prev_x, prev_y, x, y)
        prev_x, prev_y = x, y

        if collision_with_world(env, robot):
            collision_steps += 1

        if distance_xy(x, y, goal_x, goal_y) < GOAL_REACH_M:
            goals_reached += 1
            print(f"*** Reached goal #{goals_reached} at sim_t={sim_t:.2f}s — next preset goal ***")
            goal_idx = (goal_idx + 1) % len(active_goals)
            goal_x, goal_y, goal_room, path, goal_idx = first_reachable_from_index(
                env, robot, floor, (x, y), active_goals, goal_idx
            )
            replan_counter = 0
            env.task.set_start_marker(x, y, floor_z)
            env.task.set_goal_marker(goal_x, goal_y, floor_z)
            print(f"[GOAL] room={goal_room!r} target=({goal_x:.2f},{goal_y:.2f})")

        if terminated or truncated:
            og.log.warning("Episode ended unexpectedly (should not happen with DummyTask).")
            break

    if writer is not None:
        writer.release()
        print(f"[RECORD] wrote {video_path} ({step + 1} frames)")
        finalize_mp4(video_path)

    print("=" * 72)
    print("SUMMARY")
    print(f"  Sim time covered: {sim_seconds:.1f} s")
    print(f"  Goals reached (distinct waypoints): {goals_reached}")
    print(f"  Approx. path length: {total_path_m:.2f} m")
    print(f"  Steps with contact (non-floor): {collision_steps}")
    print("=" * 72)

    og.shutdown()


if __name__ == "__main__":
    main()
