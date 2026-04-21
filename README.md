# OmniGibson take-home — Part B (180 s autonomous navigation)

## System information


| Item              | Your value                                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| **OS**            | Ubuntu 22.04 (EC2)                                                                                         |
| **GPU**           | NVIDIA A10G (g5.xlarge — 1× A10G, 24 GB)                                                                   |
| **Driver / CUDA** | Driver 565.x, CUDA 12.7 (Deep Learning Base OSS Nvidia Driver GPU AMI 20250919)                            |
| **Stack**         | BEHAVIOR-1K + OmniGibson + Isaac Sim per [behavior.stanford.edu](https://behavior.stanford.edu/index.html) |


## Installation (copy-paste)

Complete the official BEHAVIOR-1K / Isaac Sim setup first. Then, from a clone that includes `BEHAVIOR-1K/OmniGibson`:

```bash
cd BEHAVIOR-1K/OmniGibson
conda activate behavior   # or the env name from the official guide
pip install -e .
```

Download scene/dataset assets so `Rs_int` loads as in Part A.

## Execution

From this repo root (`OmniGibson_TakeHomeTest`), with OmniGibson on `PYTHONPATH` (local) or on EC2 after sync:

```bash
python autonomous_nav_60s.py              # full ~180 s sim segment
python autonomous_nav_60s.py --short      # ~10 s smoke test
python autonomous_nav_60s.py --no-teleop-camera
# First-person RGB video (OpenCV mp4v, then ffmpeg H.264 baseline for QuickTime / previews):
python autonomous_nav_60s.py --record [--output output/autonomous_nav_fpv.mp4]
```

**EC2 + DCV:** Sync, then run the launcher inside the DCV desktop (see `run_partB_dcv.sh`).

```bash
chmod +x sync_to_ec2.sh dcv_tunnel.sh
./sync_to_ec2.sh --skip-behavior -i /path/to/key.pem ubuntu@<EC2_IP>
# Mac: keep tunnel open, then DCV → https://localhost:8443
./dcv_tunnel.sh -i /path/to/key.pem ubuntu@<EC2_IP>
```

On EC2 after code or asset changes, clear caches if sim behaves oddly, then restart:

```bash
rm -rf /opt/dlami/nvme/og-appdata/global/cache/*
rm -f  /opt/dlami/nvme/og-appdata/local/data/og_dataset/scenes/Rs_int/json/*.usd
```

**Recording checklist:** Terminal shows commands → launch script → autonomous motion → **180 s** continuous segment; use printed `[TIMER] sim_t=...` or an on-screen stopwatch (30 Hz × 180 s = 5400 steps).

### Benevolence multi-waypoint navigation

Extended demos visit multiple user-marked goals in **`Benevolence_1_int`** (three red-X waypoints from a marked PDF) and **`Benevolence_2_int`** (seven hard-coded waypoints x1–x7). Same stack as Part B: A* on the traversability map (with objects + doorway painting), pure pursuit, LiDAR avoidance, doors opened then non-collidable.

```bash
# Benevolence 1 — typical one-shot run with FPV recording (matches EC2 helper defaults)
python autonomous_nav_benevolence1.py --record --no-teleop-camera --once

# Benevolence 2 — seven-waypoint tour
python autonomous_nav_benevolence2.py --record --no-teleop-camera --once
```

Each script writes diagnostics under `output/` (e.g. nav CSV, events log, summary JSON, MP4 when `--record` is set). See [autonomous_nav_benevolence1.py](autonomous_nav_benevolence1.py) and [autonomous_nav_benevolence2.py](autonomous_nav_benevolence2.py).

### Waypoint path plots

Static figures overlay planned (and optionally actual) paths on the scene traversability PNG:

- [plot_marked_waypoints_path.py](plot_marked_waypoints_path.py) — detects red marks in `benevolence1_marked.pdf`, registers to `floor_trav_0.png`, runs A* between snapped waypoints. Default PDF: `docs/benevolence1_marked.pdf` in the repo, or pass `--pdf`. Output: `output/benevolence1_marked_path.png` (override with `--out`). Optional `--nav-csv` overlays the driven trajectory.
- [plot_marked_waypoints_path_benevolence2.py](plot_marked_waypoints_path_benevolence2.py) — uses hard-coded world waypoints and precomputed `output/scene_graphs/Benevolence_2_int/nav_paths.json` segments. Output: `output/benevolence2_marked_path.png`. Pass `--trav` or cache `floor_trav_0.png` under `.cache/` if `BEHAVIOR-1K/` is absent (see script docstring).

### Robot view / video renderers

Batch utilities save robot-perspective RGB (and depth for views) or short FPV videos per scene; outputs go to **`output2/`** by default.

- [render_robot_views.py](render_robot_views.py) + [render_robot_views.sh](render_robot_views.sh) — still frames per scene.
- [render_robot_videos.py](render_robot_videos.py) + [render_robot_videos.sh](render_robot_videos.sh) — FPV clips (wrapper sets conda, `OMNIGIBSON_DATA_PATH`, DCV window size, etc.).

Run the `.sh` launchers on an EC2 DCV desktop after syncing the repo; pass `--headless` for non-interactive SSH if supported by your stack.

### EC2 run / pull workflow

Helpers mirror [sync_to_ec2.sh](sync_to_ec2.sh) host resolution (`ubuntu@<EC2_IP>`, `-i /path/to/key.pem`, `EC2_SYNC_TARGET`, or `.ec2_sync_host`):

| Script | Purpose |
| ------ | ------- |
| [run_benevolence_ec2_and_pull.sh](run_benevolence_ec2_and_pull.sh) | Rsync repo, run `autonomous_nav_benevolence1.py --record --no-teleop-camera --once` on EC2 (background job), pull MP4 + logs, optional local H.264 transcode |
| [run_benevolence2_ec2_and_pull.sh](run_benevolence2_ec2_and_pull.sh) | Same for `autonomous_nav_benevolence2.py` |
| [sync_and_render_robot_videos_ec2.sh](sync_and_render_robot_videos_ec2.sh) | Sync and kick off batch robot video rendering on the instance |
| [pull_robot_videos.sh](pull_robot_videos.sh) | `rsync` `output2/robot_videos/` from EC2 to local |
| [pull_robot_views.sh](pull_robot_views.sh) | `rsync` robot view stills |
| [monitor_pull_robot_videos.sh](monitor_pull_robot_videos.sh) | Poll remote render and pull when done |
| [transcode_robot_videos_h264.sh](transcode_robot_videos_h264.sh) | After pull on macOS: re-encode `fpv_*.mp4` to H.264 for QuickTime / Finder |

For DCV tunneling from a Mac, use `./dcv_tunnel.sh -i /path/to/key.pem ubuntu@<EC2_IP>` as in **Execution** above.

## Navigation approach

- **Inputs:** 2D LiDAR from the Turtlebot `ScanSensor` (`scan` in observations, ranges in meters after denormalization); traversability grid from `InteractiveTraversableScene` / `TraversableMap` for planning; simulator **ground-truth** robot pose for position, heading, and replanning checks.
- **Localization assumption:** **Ground-truth pose** from `get_position_orientation` (no SLAM/odometry error model). LiDAR provides local obstacle geometry for reactive avoidance layered on top of the global plan.
- **Planner:** **A** shortest path on the traversability map via `env.scene.get_shortest_path(..., entire_path=True, robot=robot)`. The map uses `**trav_map_with_objects: True`** so walls are obstacles; **doorway cells** are painted from door object positions plus hardcoded `DOORWAYS` rectangles so paths can cross real openings between rooms. Replanned every *N* steps and when deviation from the path is large.
- **Controller:** **Pure pursuit** on the A polyline when a path exists (lookahead on the path → heading error → normalized differential-drive commands). If planning fails, **bearing-to-goal** (turn-then-drive) is used as fallback. Final command blends pure pursuit with **LiDAR avoidance** (sector minima: slow / stop-and-turn when forward range is short; lateral gap steers in narrow gaps). Normalized `(v, ω)` go to `DifferentialDriveController`.
- **Collision handling / safety:** **Walls** stay collidable so the robot cannot clip through room boundaries; **doors** are opened then set `**visual_only`** so panels do not narrow the doorway. LiDAR reacts to walls and furniture; optional `ContactBodies` logging ignores floors.

## Troubleshooting (what broke + how I fixed it)

- **Robot clipped through the middle wall:** Walls had been made non-collidable and the trav map ignored wall occupancy. **Fix:** Remove wall `visual_only`, set `**trav_map_with_objects: True`**, paint traversable strips at doorways, and **follow the A path with pure pursuit** instead of driving straight to the goal.
- **Stuck oscillating in doorways:** Open door meshes plus aggressive LiDAR stop/slow thresholds blocked the gap. **Fix:** Keep doors non-collidable after open, **widen doorway painting** (`DOORWAY_RADIUS_CELLS`), and **lower** `SLOW_DIST` / `STOP_DIST` so the robot commits through the opening.
- **DCV “endpoint unreachable” for `https://localhost:8443`:** The SSH **port forward** was not running. **Fix:** Run `./dcv_tunnel.sh -i key.pem ubuntu@<IP>` (or `ssh -L 8443:127.0.0.1:8443 ...`) and leave that session open while connecting.
- **Stale scene / weird load after edits:** Kit cached USD or global cache. **Fix:** Remove `og-appdata` global cache and Rs_int `json/*.usd` as in **Execution**, then restart.
- `**pkill` kills your SSH session:** Avoid `pkill -f autonomous_nav_60s.py` over SSH (the pattern matches the remote shell). Use `pkill -f '[a]utonomous_nav_60s.py'` or match only the `python` process.

## Files added in this Repo

| File | Role |
| ---- | ---- |
| [autonomous_nav_60s.py](autonomous_nav_60s.py) | Part B entry point; `--record` FPV video |
| [autonomous_nav_benevolence1.py](autonomous_nav_benevolence1.py) | Benevolence_1_int multi-waypoint nav |
| [autonomous_nav_benevolence2.py](autonomous_nav_benevolence2.py) | Benevolence_2_int seven-waypoint nav |
| [plot_marked_waypoints_path.py](plot_marked_waypoints_path.py) | Benevolence 1 path figure from marked PDF |
| [plot_marked_waypoints_path_benevolence2.py](plot_marked_waypoints_path_benevolence2.py) | Benevolence 2 path figure |
| [render_robot_views.py](render_robot_views.py), [render_robot_views.sh](render_robot_views.sh) | Per-scene robot RGB/depth stills |
| [render_robot_videos.py](render_robot_videos.py), [render_robot_videos.sh](render_robot_videos.sh) | Per-scene FPV videos |
| [run_benevolence_ec2_and_pull.sh](run_benevolence_ec2_and_pull.sh), [run_benevolence2_ec2_and_pull.sh](run_benevolence2_ec2_and_pull.sh) | EC2 run + pull for Benevolence demos |
| [sync_and_render_robot_videos_ec2.sh](sync_and_render_robot_videos_ec2.sh) | EC2 batch video render |
| [pull_robot_videos.sh](pull_robot_videos.sh), [pull_robot_views.sh](pull_robot_views.sh) | Pull `output2/` artifacts |
| [monitor_pull_robot_videos.sh](monitor_pull_robot_videos.sh) | Watch remote render + pull |
| [transcode_robot_videos_h264.sh](transcode_robot_videos_h264.sh) | Local H.264 transcode |
| [run_partB_dcv.sh](run_partB_dcv.sh), [dcv_tunnel.sh](dcv_tunnel.sh), [sync_to_ec2.sh](sync_to_ec2.sh) | Part B DCV / sync helpers |
| [cycle_scenes.sh](cycle_scenes.sh) | EC2 scene cycle launcher (`OMNIGIBSON_DATA_PATH`, etc.) |