#!/usr/bin/env python3
"""
Load each scene listed in selected_scenes.txt (or defaults) in OmniGibson / Isaac Sim.

Intended for GPU hosts (e.g. EC2 g4dn/g5). Requires omnigibson installed and BEHAVIOR-1K data.

Before importing omnigibson, this script ensures OMNIGIBSON_DATA_PATH exists (creates the default
BEHAVIOR-1K/datasets directory if unset) so omnigibson.macros can initialize.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Default curated list (matches https://behavior.stanford.edu/omnigibson/scenes.html selection)
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


def _default_data_path(script_dir: Path) -> Path:
    return (script_dir / "BEHAVIOR-1K" / "datasets").resolve()


def _prepare_runtime_env(script_dir: Path, headless: bool, data_path: Path | None) -> Path:
    """Create data dir and set env vars before importing omnigibson."""
    if data_path is None:
        data_path = _default_data_path(script_dir)
    else:
        data_path = data_path.expanduser().resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    os.environ["OMNIGIBSON_DATA_PATH"] = str(data_path)
    if headless:
        os.environ["OMNIGIBSON_HEADLESS"] = "1"
    if not os.environ.get("OMNI_KIT_ACCEPT_EULA"):
        # Common Isaac Sim / Kit expectation on headless servers
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    return data_path


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BEHAVIOR-1K assets if needed and load each scene in OmniGibson."
    )
    parser.add_argument(
        "--scenes-file",
        type=Path,
        default=Path(__file__).resolve().parent / "selected_scenes.txt",
        help="Path to newline-separated scene names (default: ./selected_scenes.txt).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="OMNIGIBSON_DATA_PATH (default: <repo>/BEHAVIOR-1K/datasets).",
    )
    parser.add_argument("--headless", action="store_true", help="Set OMNIGIBSON_HEADLESS=1.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Load only structure categories (floors, walls, ceilings, ...).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not call download_behavior_1k_assets; fail if scenes are missing.",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Accept BEHAVIOR-1K license non-interactively (required for automated download).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of env.step([]) calls after each successful load (default: 1).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path = _prepare_runtime_env(script_dir, args.headless, args.data_path)

    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import download_behavior_1k_assets
    from omnigibson.utils.constants import STRUCTURE_CATEGORIES

    print(f"OMNIGIBSON_DATA_PATH={gm.DATA_PATH}")
    if not _scenes_downloaded(data_path):
        if args.skip_download:
            sys.exit(
                f"No scenes under {_behavior_scenes_dir(data_path)}. "
                "Run without --skip-download or download the dataset manually."
            )
        if not args.accept_license:
            sys.exit(
                "Dataset missing. Re-run with --accept-license to download BEHAVIOR-1K assets "
                "non-interactively, or download manually."
            )
        print("Downloading BEHAVIOR-1K assets (this may take a long time)...")
        download_behavior_1k_assets(accept_license=True)
    else:
        print("BEHAVIOR-1K scenes directory found; skipping download.")

    scene_names = _read_scene_names(args.scenes_file)
    print(f"Loading {len(scene_names)} scene(s)...")

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for i, scene_model in enumerate(scene_names):
        cfg: dict = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_model,
            },
        }
        if args.quick:
            cfg["scene"]["load_object_categories"] = list(STRUCTURE_CATEGORIES)

        env = None
        try:
            env = og.Environment(configs=cfg)
            for _ in range(max(0, args.steps)):
                env.step([])
            successes.append(scene_model)
            print(f"OK  {scene_model}")
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
                    print(f"Warning: og.clear() failed before next scene: {clear_exc}", file=sys.stderr)

    print("\n=== Summary ===")
    print(f"Succeeded: {len(successes)}")
    print(f"Failed:    {len(failures)}")
    if failures:
        for name, err in failures:
            print(f"  - {name}: {err}")

    og.shutdown()


if __name__ == "__main__":
    main()
