#!/usr/bin/env python3
"""
Capture RGB (and optional depth) images from a Turtlebot VisionSensor across BEHAVIOR-1K scenes.

Loads each scene with full object content, samples interior-biased traversable poses (away from walls),
orients the robot toward nearby furniture/objects, stabilizes, and saves robot-centric views under
output2/robot_views/<scene_name>/ by default.

Requires GPU Isaac Sim / OmniGibson (same stack as load_scenes.py / autonomous_nav_60s.py).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch as th

# Repo layout: OmniGibson_TakeHomeTest/BEHAVIOR-1K/OmniGibson/omnigibson
_REPO_ROOT = Path(__file__).resolve().parent
_OG_SRC = _REPO_ROOT / "BEHAVIOR-1K" / "OmniGibson"
if _OG_SRC.is_dir() and str(_OG_SRC) not in sys.path:
    sys.path.insert(0, str(_OG_SRC))

DEFAULT_SCENE_NAMES = [
    "Benevolence_0_int",
    "Benevolence_1_int",
    "Benevolence_2_int",
    "Merom_0_int",
    "Pomaria_0_int",
    "Rs_int",
    "Beechwood_0_int",
    "Beechwood_1_int",
    "house_double_floor_lower",
    "house_double_floor_upper",
]

POST_SETTLE_SIM_STEPS = 20
POST_SETTLE_RENDER_PASSES = 5
ROBOT_NAME = "view_turtlebot"
ACTION_FREQ = 30
# Extra erosion beyond robot footprint to stay away from walls (meters).
DEFAULT_INTERIOR_MARGIN_M = 0.9


def _prepare_runtime_env(*, headless: bool, data_path: Path | None) -> None:
    """Set env vars before importing omnigibson. Only set OMNIGIBSON_DATA_PATH if --data-path."""
    if data_path is not None:
        resolved = data_path.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ["OMNIGIBSON_DATA_PATH"] = str(resolved)
    if headless:
        os.environ["OMNIGIBSON_HEADLESS"] = "1"
    if not os.environ.get("OMNI_KIT_ACCEPT_EULA"):
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def _behavior_scenes_dir(data_path: Path) -> Path:
    return data_path / "behavior-1k-assets" / "scenes"


def _scenes_downloaded(data_path: Path) -> bool:
    p = _behavior_scenes_dir(data_path)
    if not p.is_dir():
        return False
    return any(
        x.is_dir() and not x.name.startswith(".") and x.name != "background" for x in p.iterdir()
    )


def _read_scene_names(scenes_file: Path) -> list[str]:
    if not scenes_file.is_file():
        print(f"Scenes file not found: {scenes_file}, using built-in defaults.", file=sys.stderr)
        return list(DEFAULT_SCENE_NAMES)
    names: list[str] = []
    for line in scenes_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            names.append(s)
    return names if names else list(DEFAULT_SCENE_NAMES)


def choose_floor_index(scene_name: str, n_floors: int) -> int | None:
    """
    Pick trav-map floor for sampling. None = random floor each call (multi-floor diversity).
    """
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
    # Beechwood / other multi-floor: randomize per sample
    return None


def build_env_config(scene_model: str, resolution: int, structure_only: bool) -> dict:
    """Environment with Turtlebot + DummyTask; full scene objects unless structure_only."""
    scene_cfg: dict = {
        "type": "InteractiveTraversableScene",
        "scene_model": scene_model,
        "trav_map_resolution": 0.1,
        "default_erosion_radius": 0.0,
        "trav_map_with_objects": True,
        "num_waypoints": 1,
        "waypoint_resolution": 0.2,
        "scene_source": "OG",
        "include_robots": True,
    }
    if structure_only:
        from omnigibson.utils.constants import STRUCTURE_CATEGORIES

        scene_cfg["load_object_categories"] = list(STRUCTURE_CATEGORIES)

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
        "scene": scene_cfg,
        "robots": [
            {
                "type": "Turtlebot",
                "name": ROBOT_NAME,
                "obs_modalities": ["rgb", "depth"],
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": resolution,
                            "image_width": resolution,
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
    """Find tensor/array for @key in nested per-sensor robot observation dict."""
    for _sensor_name, sensor_obs in robot_obs.items():
        if isinstance(sensor_obs, dict) and key in sensor_obs:
            return sensor_obs[key]
    return None


def tensor_to_numpy(x):
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_rgb_png(path: Path, rgb_tensor) -> None:
    """Save RGB observation (H,W,4) or (H,W,3) uint8/float to PNG."""
    from PIL import Image

    arr = tensor_to_numpy(rgb_tensor)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected HxWx>=3 RGB image, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def save_depth_png(path: Path, depth_tensor) -> None:
    """Save depth map as grayscale PNG (percentile-normalized for visibility)."""
    from PIL import Image

    d = tensor_to_numpy(depth_tensor)
    if d.ndim != 2:
        d = np.squeeze(d)
    if d.ndim != 2:
        raise ValueError(f"Expected HxW depth, got shape {d.shape}")
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        vmin = float(np.percentile(d[valid], 5))
        vmax = float(np.percentile(d[valid], 95))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = (d - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(d, dtype=np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, mode="L").save(path)


def stabilize_and_grab_obs(env, robot_name: str):
    """Physics settle + render refresh, then observations."""
    import omnigibson as og

    zero = th.tensor([0.0, 0.0], dtype=th.float32)
    for _ in range(POST_SETTLE_SIM_STEPS):
        env.step({robot_name: zero})
    for _ in range(POST_SETTLE_RENDER_PASSES):
        og.sim.render()
    return env.get_obs()


def _resolve_floor_for_sample(scene_name: str, n_floors: int, floor_arg: int | None) -> int:
    """Single floor index for one pose sample (random floor when multi-floor diversity is enabled)."""
    if floor_arg is not None:
        return int(floor_arg)
    mode = choose_floor_index(scene_name, n_floors)
    if mode is not None:
        return int(mode)
    return int(th.randint(0, n_floors, (1,)).item())


def _erode_trav_map_pixels(trav_map_uint8: np.ndarray, map_resolution: float, radius_m: float) -> np.ndarray:
    """Binary erosion in map pixels; @trav_map_uint8 is 0/255."""
    radius_px = max(1, int(math.ceil(radius_m / max(map_resolution, 1e-6))))
    kernel = np.ones((radius_px, radius_px), dtype=np.uint8)
    return cv2.erode(trav_map_uint8.astype(np.uint8), kernel)


def yaw_toward_nearest_content_object(env, robot, rx: float, ry: float) -> float | None:
    """
    Bearing (world frame) to face the nearest non-structure object from (rx, ry).
    Returns None if no suitable target exists.
    """
    from omnigibson.robots.robot_base import m as robot_macros
    from omnigibson.utils.constants import STRUCTURAL_DOOR_CATEGORIES, STRUCTURE_CATEGORIES

    best_d2 = float("inf")
    best: tuple[float, float] | None = None
    min_dist2 = 0.15 * 0.15  # ignore targets almost on top of the robot

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


def try_land_diverse_pose(
    env,
    robot,
    scene_name: str,
    floor_arg: int | None,
    max_tries: int,
    interior_margin_m: float,
) -> bool:
    """
    Sample traversable xy biased toward room interiors (heavy map erosion), yaw toward nearest
    furniture/object; land if valid pose.
    """
    import omnigibson.utils.transform_utils as T
    from omnigibson.utils.sim_utils import land_object, test_valid_pose

    z_off = env.initial_pos_z_offset
    trav = env.scene.trav_map
    n_floors = trav.n_floors

    for _ in range(max_tries):
        floor = _resolve_floor_for_sample(scene_name, n_floors, floor_arg)
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
            continue

        j = int(th.randint(0, free_r.numel(), (1,)).item())
        r_i, c_i = int(free_r[j].item()), int(free_c[j].item())
        xy_map = th.tensor([float(r_i), float(c_i)], dtype=th.float32)
        wxy = trav.map_to_world(xy_map)
        rx = float(wxy[0].item())
        ry = float(wxy[1].item())
        fh = trav.floor_heights[floor]
        rz = float(fh.item()) if isinstance(fh, th.Tensor) else float(fh)
        rpos = th.tensor([rx, ry, rz], dtype=th.float32)

        yaw = yaw_toward_nearest_content_object(env, robot, rx, ry)
        if yaw is None:
            yaw = (th.rand(1) * 2.0 * math.pi).item()
        quat = T.euler2quat(th.tensor([0.0, 0.0, yaw]))
        if test_valid_pose(robot, rpos, quat, z_off):
            land_object(robot, rpos, quat, z_off)
            return True
    return False


def run_dry_run_save(output_dir: Path) -> None:
    """Write synthetic PNGs to verify PIL paths (no simulator)."""
    out = output_dir / "_dry_run"
    rgb = np.zeros((64, 64, 4), dtype=np.uint8)
    rgb[:, :, 0] = 200
    rgb[:, :, 1] = 100
    rgb[:, :, 2] = 50
    rgb[:, :, 3] = 255
    save_rgb_png(out / "rgb_synthetic.png", rgb)
    depth = np.linspace(0.5, 5.0, 4096, dtype=np.float32).reshape(64, 64)
    save_depth_png(out / "depth_synthetic.png", depth)
    assert (out / "rgb_synthetic.png").is_file()
    assert (out / "depth_synthetic.png").is_file()
    print(f"Dry-run OK: wrote {out / 'rgb_synthetic.png'} and depth_synthetic.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save robot-perspective RGB/depth images for each BEHAVIOR-1K scene."
    )
    parser.add_argument(
        "--scenes-file",
        type=Path,
        default=_REPO_ROOT / "selected_scenes.txt",
        help="Newline-separated scene names (default: ./selected_scenes.txt).",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Comma-separated scene names (overrides --scenes-file if set).",
    )
    parser.add_argument("--resolution", type=int, default=512, help="VisionSensor width/height (default: 512).")
    parser.add_argument("--num-views", type=int, default=3, help="Random poses per scene (default: 3).")
    parser.add_argument("--headless", action="store_true", help="Set OMNIGIBSON_HEADLESS=1.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="OMNIGIBSON_DATA_PATH (default: <repo>/BEHAVIOR-1K/datasets).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "output2" / "robot_views",
        help="Output root (default: output2/robot_views).",
    )
    parser.add_argument(
        "--interior-margin",
        type=float,
        default=DEFAULT_INTERIOR_MARGIN_M,
        help=(
            "Extra traversability erosion radius in meters to stay away from walls "
            f"(default: {DEFAULT_INTERIOR_MARGIN_M})."
        ),
    )
    parser.add_argument("--skip-download", action="store_true", help="Fail if scene assets are missing.")
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Accept BEHAVIOR-1K license for non-interactive download.",
    )
    parser.add_argument("--skip-depth", action="store_true", help="Only save RGB images.")
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Load only structure categories (faster, less photorealistic).",
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=None,
        help="Force traversability floor index (default: infer from scene name / random multi-floor).",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible poses.")
    parser.add_argument(
        "--pose-tries",
        type=int,
        default=100,
        help="Max random pose attempts per view before skipping (default: 100).",
    )
    parser.add_argument(
        "--dry-run-save",
        action="store_true",
        help="Test PNG writers only (no OmniGibson); writes under output-dir/_dry_run.",
    )
    args = parser.parse_args()

    if args.dry_run_save:
        run_dry_run_save(args.output_dir.expanduser().resolve())
        return

    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)

    _prepare_runtime_env(headless=args.headless, data_path=args.data_path)

    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import download_behavior_1k_assets

    data_path = Path(gm.DATA_PATH)
    print(f"OMNIGIBSON_DATA_PATH={gm.DATA_PATH}")

    if not _scenes_downloaded(data_path):
        if args.skip_download:
            sys.exit(
                f"No scenes under {_behavior_scenes_dir(data_path)}. "
                "Run without --skip-download or download assets manually."
            )
        if not args.accept_license:
            sys.exit(
                "Dataset missing. Re-run with --accept-license to download BEHAVIOR-1K assets, "
                "or download manually."
            )
        print("Downloading BEHAVIOR-1K assets (may take a long time)...")
        download_behavior_1k_assets(accept_license=True)
    else:
        print("BEHAVIOR-1K scenes directory found; skipping download.")

    if args.scenes:
        scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scene_names = _read_scene_names(args.scenes_file.expanduser().resolve())

    out_root = args.output_dir.expanduser().resolve()
    print(f"Saving to {out_root} ({len(scene_names)} scene(s), {args.num_views} view(s) each)")

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for i, scene_model in enumerate(scene_names):
        env = None
        try:
            cfg = build_env_config(scene_model, args.resolution, args.structure_only)
            env = og.Environment(configs=cfg)
            obs, _ = env.reset()
            robot = env.robots[0]
            assert robot.name == ROBOT_NAME

            scene_dir = out_root / scene_model
            saved = 0
            for v in range(args.num_views):
                if not try_land_diverse_pose(
                    env,
                    robot,
                    scene_model,
                    args.floor,
                    args.pose_tries,
                    args.interior_margin,
                ):
                    print(f"  [{scene_model}] view {v}: could not find valid pose, skipping")
                    continue
                obs, _ = stabilize_and_grab_obs(env, ROBOT_NAME)
                rob_obs = obs[ROBOT_NAME]
                rgb = extract_modality_from_obs(rob_obs, "rgb")
                if rgb is None:
                    print(f"  [{scene_model}] view {v}: no rgb in obs keys={list(rob_obs.keys())!r}")
                    continue
                save_rgb_png(scene_dir / f"rgb_{saved}.png", rgb)
                if not args.skip_depth:
                    depth = extract_modality_from_obs(rob_obs, "depth")
                    if depth is not None:
                        save_depth_png(scene_dir / f"depth_{saved}.png", depth)
                    else:
                        print(f"  [{scene_model}] view {v}: depth missing (skip-depth not set)")
                print(f"  [{scene_model}] saved view {saved} -> rgb_{saved}.png")
                saved += 1

            if saved > 0:
                successes.append(scene_model)
                print(f"OK  {scene_model}  ({saved} image set(s))")
            else:
                failures.append((scene_model, "no views saved"))
                print(f"FAIL {scene_model}: no views saved")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            failures.append((scene_model, err))
            print(f"FAIL {scene_model}: {err}")
            traceback.print_exc()
        finally:
            del env
            if i < len(scene_names) - 1 and og.sim is not None:
                try:
                    og.clear()
                except Exception as clear_exc:
                    print(f"Warning: og.clear() failed: {clear_exc}", file=sys.stderr)

    print("\n=== Summary ===")
    print(f"Succeeded: {len(successes)}")
    print(f"Failed:    {len(failures)}")
    if failures:
        for name, err in failures:
            print(f"  - {name}: {err}")

    og.shutdown()


if __name__ == "__main__":
    main()
