#!/usr/bin/env python3
"""
List iGibson / BEHAVIOR scenes on disk and pick a recommended subset.
Filesystem only — no simulator or environment loading.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Only count traversability maps with objects (not floor_trav_no_obj_*.png)
_FLOOR_TRAV_RE = re.compile(r"^floor_trav_(\d+)\.png$", re.IGNORECASE)

# Curated indoor mix when no dataset is on disk (see BEHAVIOR OmniGibson scenes docs).
CURATED_SCENE_NAMES: list[str] = [
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
CURATED_MULTI_FLOOR = frozenset(
    {
        "Beechwood_0_int",
        "Beechwood_1_int",
        "house_double_floor_lower",
        "house_double_floor_upper",
    }
)


def curated_scene_rows() -> list[tuple[str, bool]]:
    return [(name, name in CURATED_MULTI_FLOOR) for name in CURATED_SCENE_NAMES]


def _is_scene_entry(name: str) -> bool:
    return not name.startswith(".") and name != "background"


def _count_floor_trav_maps(layout_dir: Path) -> int:
    if not layout_dir.is_dir():
        return 0
    n = 0
    for p in layout_dir.iterdir():
        if p.is_file() and _FLOOR_TRAV_RE.match(p.name):
            n += 1
    return n


def _try_igibson_scenes_dir() -> Path | None:
    try:
        from igibson.utils.assets_utils import get_ig_assets_path  # type: ignore
    except ImportError:
        return None
    try:
        root = Path(get_ig_assets_path()).expanduser().resolve()
    except Exception:
        return None
    scenes = root / "ig_dataset" / "scenes"
    return scenes if scenes.is_dir() else None


def _default_behavior_datasets_root(script_dir: Path) -> Path:
    """BEHAVIOR-1K repo layout: <workspace>/BEHAVIOR-1K/datasets"""
    return (script_dir / "BEHAVIOR-1K" / "datasets").resolve()


def _try_omnigibson_scenes_dir(script_dir: Path) -> Path | None:
    env = os.environ.get("OMNIGIBSON_DATA_PATH")
    if env:
        data_path = Path(os.path.expanduser(env)).resolve()
    else:
        data_path = _default_behavior_datasets_root(script_dir)
    if not data_path.is_dir():
        return None
    scenes = data_path / "behavior-1k-assets" / "scenes"
    return scenes if scenes.is_dir() else None


def discover_scenes_dir(explicit: Path | None, script_dir: Path) -> Path | None:
    """Return scenes root, or None to use curated fallback (no dataset on disk)."""
    if explicit is not None:
        if not explicit.is_dir():
            sys.exit(f"Scenes directory does not exist: {explicit}")
        return explicit.resolve()

    ig = _try_igibson_scenes_dir()
    if ig is not None:
        return ig

    og = _try_omnigibson_scenes_dir(script_dir)
    if og is not None:
        return og

    return None


def collect_scene_info(scenes_dir: Path) -> list[tuple[str, bool]]:
    rows: list[tuple[str, bool]] = []
    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_dir() or not _is_scene_entry(entry.name):
            continue
        n_maps = _count_floor_trav_maps(entry / "layout")
        # No layout or zero maps: treat as single-floor (unknown / missing data)
        multi = n_maps > 1
        rows.append((entry.name, multi))
    return rows


def select_scenes(rows: list[tuple[str, bool]]) -> list[str]:
    """Up to 10 scenes: prefer 3 multi-floor + 7 single-floor; backfill with singles."""
    if not rows:
        return []

    names_sorted = sorted(name for name, _ in rows)
    if len(names_sorted) <= 10:
        return names_sorted

    multi = sorted(name for name, m in rows if m)
    single = sorted(name for name, m in rows if not m)

    take_multi = multi[:3]
    need_single = 10 - len(take_multi)
    take_single = single[:need_single]

    # If not enough singles to fill (unusual), add more multis
    selected = take_single + take_multi
    if len(selected) < 10:
        used = set(selected)
        for name in multi:
            if len(selected) >= 10:
                break
            if name not in used:
                selected.append(name)
                used.add(name)
    return selected[:10]


def print_table(rows: list[tuple[str, bool]]) -> None:
    if not rows:
        print("No scenes found.")
        return
    col0 = max(len(r[0]) for r in rows)
    header = f"{'Scene Name'.ljust(col0)}  Multi-Floor"
    print(header)
    print("-" * len(header))
    for name, multi in sorted(rows, key=lambda x: x[0].lower()):
        yn = "yes" if multi else "no"
        print(f"{name.ljust(col0)}  {yn}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List scenes under ig_dataset/scenes or behavior-1k-assets/scenes."
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=None,
        help="Explicit path to the directory containing scene folders (overrides auto-discovery).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("selected_scenes.txt"),
        help="Where to write selected scene names (default: ./selected_scenes.txt).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    scenes_dir = discover_scenes_dir(args.scenes_dir, script_dir)

    if scenes_dir is None:
        print(
            "No scenes directory found (iGibson ig_dataset or BEHAVIOR-1K behavior-1k-assets/scenes).\n"
            "Using curated fallback list from BEHAVIOR OmniGibson scenes documentation.\n"
            "Install data or pass --scenes-dir PATH to scan real assets.\n"
        )
        rows = curated_scene_rows()
        selected = list(CURATED_SCENE_NAMES)
    else:
        rows = collect_scene_info(scenes_dir)
        selected = select_scenes(rows)

    print(f"Scenes directory: {scenes_dir or '(none — curated fallback)'}\n")
    print_table(rows)

    multi_set = {name for name, m in rows if m}

    print("\n--- Recommended selection (10 scenes when possible) ---")
    if not selected:
        print("(none)")
    else:
        singles_picked = [n for n in selected if n not in multi_set]
        multis_picked = [n for n in selected if n in multi_set]
        print(f"Single-floor ({len(singles_picked)}): {', '.join(singles_picked) or '(none)'}")
        print(f"Multi-floor ({len(multis_picked)}): {', '.join(multis_picked) or '(none)'}")
        print("\nFinal list (in order written to file):")
        for name in selected:
            print(f"  - {name}")

    args.output.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    print(f"\nWrote {len(selected)} scene(s) to {args.output.resolve()}")


if __name__ == "__main__":
    main()
