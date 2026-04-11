#!/usr/bin/env python3
"""
Cycle through 10 hardcoded BEHAVIOR-1K scenes in OmniGibson, ~N simulated seconds each.

Does not read selected_scenes.txt — scene names are fixed in SCENE_NAMES below.

By default does not set OMNIGIBSON_DATA_PATH (same as autonomous_nav_60s.py): OmniGibson
resolves datasets via omnigibson.macros (typically BEHAVIOR-1K/datasets next to the install).
Use --data-path only to override.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# Curated indoor mix (BEHAVIOR OmniGibson scenes); order is fixed.
SCENE_NAMES: list[str] = [
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

# Match omnigibson.macros.gm.DEFAULT_SIM_STEP_FREQ (env default action_frequency).
ACTION_FREQ_HZ = 30


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cycle hardcoded BEHAVIOR-1K scenes for N simulated seconds each."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override OMNIGIBSON_DATA_PATH (default: OmniGibson install default, e.g. BEHAVIOR-1K/datasets).",
    )
    parser.add_argument("--headless", action="store_true", help="Set OMNIGIBSON_HEADLESS=1.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load all interactive objects (default: structure categories only, faster).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Simulated seconds per scene (default: 5).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download BEHAVIOR-1K assets; exit if scenes are missing.",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Accept BEHAVIOR-1K license for non-interactive download.",
    )
    args = parser.parse_args()

    if args.seconds < 0:
        sys.exit("--seconds must be non-negative")

    _prepare_runtime_env(headless=args.headless, data_path=args.data_path)

    import omnigibson as og
    from omnigibson.utils.asset_utils import download_behavior_1k_assets
    from omnigibson.utils.constants import STRUCTURE_CATEGORIES

    from omnigibson.macros import gm

    data_path = Path(gm.DATA_PATH)
    print(f"OMNIGIBSON_DATA_PATH={gm.DATA_PATH}")

    if not _scenes_downloaded(data_path):
        if args.skip_download:
            sys.exit(
                f"No scenes under {_behavior_scenes_dir(data_path)}. "
                "Run without --skip-download or download manually."
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

    steps_per_scene = max(0, int(round(args.seconds * ACTION_FREQ_HZ)))
    print(
        f"Cycling {len(SCENE_NAMES)} scene(s), ~{args.seconds} s sim each "
        f"({steps_per_scene} steps @ {ACTION_FREQ_HZ} Hz).\n"
    )

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for i, scene_model in enumerate(SCENE_NAMES):
        cfg: dict = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_model,
            },
        }
        if not args.full:
            cfg["scene"]["load_object_categories"] = list(STRUCTURE_CATEGORIES)

        env = None
        t0 = time.perf_counter()
        try:
            env = og.Environment(configs=cfg)
            for _ in range(steps_per_scene):
                env.step([])
            elapsed = time.perf_counter() - t0
            successes.append(scene_model)
            print(f"OK  {scene_model}  (wall {elapsed:.1f}s, {steps_per_scene} steps)")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            failures.append((scene_model, err))
            print(f"FAIL {scene_model}: {err}")
            traceback.print_exc()
        finally:
            del env
            if i < len(SCENE_NAMES) - 1 and og.sim is not None:
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
