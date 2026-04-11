#!/usr/bin/env python3
"""
Regenerate scene_graph_overlay_floor_*.png and birds_eye_layout_floor_*.png
for scenes whose output/scene_graphs/<scene>/scene_graph.json was edited by hand.

Does not modify build_scene_graphs.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx

from build_scene_graphs import (
    MAP_DEFAULT_RESOLUTION,
    draw_birds_eye_layout,
    draw_overlays,
    find_resolution_m_per_pixel,
)

SCENES = (
    "house_double_floor_lower",
    "house_double_floor_upper",
    "Merom_0_int",
)


def envelope_to_graph(env: dict) -> nx.Graph:
    data = env["graph"]
    return nx.node_link_graph(data, edges="links", multigraph=False)


def main() -> int:
    repo = Path(__file__).resolve().parent
    scenes_root = repo / "BEHAVIOR-1K" / "datasets" / "behavior-1k-assets" / "scenes"
    out_base = repo / "output" / "scene_graphs"

    if not scenes_root.is_dir():
        print(f"error: scenes root not found: {scenes_root}", file=sys.stderr)
        return 1

    for scene in SCENES:
        json_path = out_base / scene / "scene_graph.json"
        if not json_path.is_file():
            print(f"error: missing {json_path}", file=sys.stderr)
            return 1
        with open(json_path, encoding="utf-8") as f:
            env = json.load(f)
        G = envelope_to_graph(env)
        scene_dir = scenes_root / scene
        layout_dir = scene_dir / "layout"
        out_dir = out_base / scene
        res = find_resolution_m_per_pixel(scene_dir)
        if res is None:
            res = MAP_DEFAULT_RESOLUTION
            print(f"  [info] {scene}: using overlay resolution {res} m/pixel (no JSON metadata)")
        else:
            print(f"  [info] {scene}: overlay resolution {res} m/pixel from scene metadata")
        draw_overlays(G, layout_dir, scene, out_dir, res)
        draw_birds_eye_layout(G, layout_dir, scene, out_dir, res, scenes_root)
        print(f"  done: {scene}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
