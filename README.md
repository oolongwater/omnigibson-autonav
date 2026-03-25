# OmniGibson take-home — Part B (180 s autonomous navigation)

## System information

Fill in for your submission (or paste output of `nvidia-smi` on the machine that runs Isaac Sim):


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

`[autonomous_nav_60s.py](autonomous_nav_60s.py)` — Part B entry point. `[run_partB_dcv.sh](run_partB_dcv.sh)`, `[dcv_tunnel.sh](dcv_tunnel.sh)`, `[sync_to_ec2.sh](sync_to_ec2.sh)` — remote workflow helpers.