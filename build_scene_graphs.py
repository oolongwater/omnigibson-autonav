#!/usr/bin/env python3
"""
Build navigation-focused structural scene graphs from BEHAVIOR-1K scene JSONs.
No OmniGibson / Isaac Sim — filesystem + JSON only.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

# Default traversability PNG resolution (meters per pixel), per OmniGibson TraversableMap
MAP_DEFAULT_RESOLUTION = 0.01

_FLOOR_TRAV_RE = re.compile(r"^floor_trav_(\d+)\.png$", re.IGNORECASE)

# Curated fallback if selected_scenes.txt is missing
DEFAULT_SCENE_LIST = [
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

RESOLUTION_KEYS = (
    "trav_map_resolution",
    "resolution",
    "map_resolution",
    "seg_map_resolution",
)

# Native resolution of floor_insseg / floor_semseg PNGs (OmniGibson SegmentationMap default)
SEG_MAP_NATIVE_MPP = 0.01

# Proximity heuristics when segmentation is missing or inconclusive (meters)
DOOR_PROXIMITY_MAX_M = 4.5
ISOLATED_ROOM_LINK_MAX_M = 4.0

# Overlay: crop margin around building footprint (pixels)
OVERLAY_CROP_MARGIN_PX = 28

# When trav-map bbox area exceeds this ratio times the graph-node bbox area, crop to node extent
# Must exceed ~19 when trav bbox is huge vs node span (wrong mpp inflates this; default is 0.01).
# house_double_floor_lower still ~109x and uses node crop.
CROP_TRAV_VS_NODE_AREA_RATIO = 20.0

# Graph-node crop margin (pixels) when using node-based bbox
GRAPH_NODE_CROP_MARGIN_PX = 100


def get_object_registry(state: dict[str, Any]) -> dict[str, Any]:
    """Support state.registry.object_registry and legacy state.object_registry."""
    if not isinstance(state, dict):
        return {}
    reg = state.get("registry")
    if isinstance(reg, dict) and "object_registry" in reg:
        or_ = reg.get("object_registry")
        return or_ if isinstance(or_, dict) else {}
    or_ = state.get("object_registry")
    return or_ if isinstance(or_, dict) else {}


def load_scene_objects(json_path: Path) -> list[dict[str, Any]]:
    """
    Returns a list of dicts, one per object:
      name, category, in_rooms, pos, scale
    Prints warnings for missing expected keys (does not crash).
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    objects_info = data.get("objects_info") or {}
    init_info = objects_info.get("init_info")
    if not isinstance(init_info, dict):
        print(f"  [warn] {json_path}: missing or invalid objects_info.init_info")
        return []

    state = data.get("state")
    object_registry = get_object_registry(state if isinstance(state, dict) else {})

    rows: list[dict[str, Any]] = []
    for name, obj in init_info.items():
        if not isinstance(obj, dict):
            print(f"  [warn] object {name!r}: not a dict, skipping")
            continue
        args = obj.get("args")
        if not isinstance(args, dict):
            print(f"  [warn] object {name!r}: missing args dict")
            continue

        missing = [k for k in ("category",) if k not in args]
        if missing:
            print(f"  [warn] object {name!r}: missing keys in args: {missing}")

        category = args.get("category")
        if category is None:
            category = "unknown"

        in_rooms = args.get("in_rooms")
        if in_rooms is None:
            in_rooms = []
        elif isinstance(in_rooms, str):
            in_rooms = [in_rooms]
        elif not isinstance(in_rooms, list):
            print(f"  [warn] object {name!r}: in_rooms not list/str, coercing to []")
            in_rooms = []

        scale = args.get("scale")
        if scale is None:
            scale = None
        elif not (isinstance(scale, (list, tuple)) and len(scale) >= 3):
            print(f"  [warn] object {name!r}: unexpected scale {scale!r}")
            scale = None

        pos = None
        ent = object_registry.get(name)
        if isinstance(ent, dict):
            rl = ent.get("root_link")
            if isinstance(rl, dict) and "pos" in rl:
                pos = rl.get("pos")
                if not (isinstance(pos, (list, tuple)) and len(pos) >= 3):
                    print(f"  [warn] object {name!r}: invalid root_link.pos")
                    pos = None
            else:
                print(f"  [warn] object {name!r}: no root_link.pos in object_registry")
        else:
            print(f"  [warn] object {name!r}: not in object_registry / no state entry")

        rows.append(
            {
                "name": str(name),
                "category": str(category),
                "in_rooms": [str(r) for r in in_rooms if str(r).strip()],
                "pos": list(pos) if pos is not None else None,
                "scale": list(scale) if scale is not None else None,
            }
        )

    return rows


def parse_room_name(room_id: str) -> tuple[str, int]:
    """Split 'living_room_0' -> ('living_room', 0). Fallback: (room_id, 0)."""
    m = re.match(r"^(.*)_(\d+)$", room_id)
    if m:
        return m.group(1), int(m.group(2))
    return room_id, 0


def count_floor_trav_maps(layout_dir: Path) -> int:
    if not layout_dir.is_dir():
        return 0
    indices: set[int] = set()
    for p in layout_dir.iterdir():
        if p.is_file():
            m = _FLOOR_TRAV_RE.match(p.name)
            if m:
                indices.add(int(m.group(1)))
    return len(indices) if indices else 0


def mean_z_for_room(objects: list[dict[str, Any]], room_id: str) -> float | None:
    zs = []
    for o in objects:
        if room_id not in o["in_rooms"]:
            continue
        p = o["pos"]
        if p is not None and len(p) >= 3:
            zs.append(float(p[2]))
    if not zs:
        return None
    return float(np.mean(zs))


def assign_room_floor_ids_v2(
    room_ids: list[str], objects: list[dict[str, Any]], num_floors: int
) -> dict[str, int]:
    """Assign floor_id using mean Z quantile bins."""
    if num_floors <= 1:
        return {r: 0 for r in room_ids}

    room_z: dict[str, float] = {}
    for r in room_ids:
        mz = mean_z_for_room(objects, r)
        if mz is not None:
            room_z[r] = mz

    if not room_z:
        return {r: 0 for r in room_ids}

    values = np.array(list(room_z.values()), dtype=float)
    # Bin edges: min to max split into num_floors intervals
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-6:
        return {r: 0 for r in room_ids}

    edges = np.linspace(vmin, vmax, num_floors + 1)
    out: dict[str, int] = {}
    for r in room_ids:
        if r not in room_z:
            out[r] = 0
            continue
        z = room_z[r]
        # digitize: bins are edges[1], edges[2], ...
        idx = int(np.digitize(z, edges[1:-1], right=False))
        idx = max(0, min(num_floors - 1, idx))
        out[r] = idx
    return out


def is_door_category(category: str) -> bool:
    c = category.lower()
    return "door" in c


def is_stair_category(category: str) -> bool:
    return category.lower() in ("staircase", "stairs")


def infer_door_floor_id(
    connected_rooms: list[str], room_floor: dict[str, int]
) -> int:
    if not connected_rooms:
        return 0
    floors = [room_floor.get(r, 0) for r in connected_rooms]
    if len(set(floors)) == 1:
        return floors[0]
    return min(floors)


def find_resolution_m_per_pixel(scene_dir: Path) -> float | None:
    """Search scene JSON for trav_map_resolution / map_resolution-style keys."""
    json_dir = scene_dir / "json"
    if not json_dir.is_dir():
        return None
    for path in json_dir.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for key in RESOLUTION_KEYS:
            if key in data and isinstance(data[key], (int, float)):
                return float(data[key])
        init_info = data.get("init_info")
        if isinstance(init_info, dict):
            args = init_info.get("args")
            if isinstance(args, dict):
                for key in RESOLUTION_KEYS:
                    if key in args and isinstance(args[key], (int, float)):
                        return float(args[key])
    return None


def world_xy_to_pixel(
    wx: float, wy: float, img_w: int, img_h: int, resolution: float
) -> tuple[int, int]:
    """Match OmniGibson BaseMap: col = wx/r + W/2, row = wy/r + H/2 (then integer)."""
    col = int(round(wx / resolution + img_w / 2.0))
    row = int(round(wy / resolution + img_h / 2.0))
    return col, row


def find_room_categories_path(scenes_root: Path) -> Path | None:
    """behavior-1k-assets/metadata/room_categories.txt next to scenes/."""
    p = scenes_root.parent / "metadata" / "room_categories.txt"
    return p if p.is_file() else None


def load_ins_sem_arrays(
    layout_dir: Path, floor_idx: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load floor_insseg_N.png and floor_semseg_N.png as uint8 HxW; try floor 0 if N missing."""
    for idx in (floor_idx, 0):
        ins_p = layout_dir / f"floor_insseg_{idx}.png"
        sem_p = layout_dir / f"floor_semseg_{idx}.png"
        if ins_p.is_file() and sem_p.is_file():
            try:
                ins = np.asarray(Image.open(ins_p).convert("L"))
                sem = np.asarray(Image.open(sem_p).convert("L"))
                if ins.shape == sem.shape:
                    return ins, sem
            except OSError:
                pass
    return None, None


def build_sem_id_to_name(room_categories_path: Path) -> dict[int, str]:
    lines = [
        ln.rstrip()
        for ln in room_categories_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    return {i + 1: lines[i] for i in range(len(lines))}


def build_ins_id_to_room_from_sem(
    ins_arr: np.ndarray,
    sem_arr: np.ndarray,
    sem_id_to_name: dict[int, str],
) -> dict[int, str]:
    """Match OmniGibson SegmentationMap: group instance IDs by semantic ID, name as type_i."""
    sem_id_to_ins_ids: dict[int, list[int]] = defaultdict(list)
    for ins_id in np.unique(ins_arr):
        ins_id = int(ins_id)
        if ins_id == 0:
            continue
        ys, xs = np.where(ins_arr == ins_id)
        if ys.size == 0:
            continue
        sem_id = int(sem_arr[ys[0], xs[0]])
        if sem_id == 0:
            continue
        sem_id_to_ins_ids[sem_id].append(ins_id)

    ins_to_room: dict[int, str] = {}
    for sem_id, ins_ids in sem_id_to_ins_ids.items():
        sem_name = sem_id_to_name.get(sem_id)
        if not sem_name:
            continue
        ins_ids = sorted(ins_ids)
        for i, ins_id in enumerate(ins_ids):
            ins_to_room[ins_id] = f"{sem_name}_{i}"
    return ins_to_room


def bootstrap_ins_to_room_from_centroids(
    ins_arr: np.ndarray,
    room_ids: list[str],
    room_xy: dict[str, list[float]],
    img_w: int,
    img_h: int,
    resolution: float,
) -> dict[int, str]:
    """Map instance segmentation IDs to room instance names using room centroids."""

    def nearest_nonzero(r0: int, c0: int, max_r: int = 24) -> int:
        h, w = ins_arr.shape
        for rad in range(max_r + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < h and 0 <= c < w:
                        v = int(ins_arr[r, c])
                        if v != 0:
                            return v
        return 0

    ins_to_room: dict[int, str] = {}
    for rid in room_ids:
        cx, cy = room_xy[rid][0], room_xy[rid][1]
        col, row = world_xy_to_pixel(cx, cy, img_w, img_h, resolution)
        if not (0 <= row < ins_arr.shape[0] and 0 <= col < ins_arr.shape[1]):
            continue
        val = int(ins_arr[row, col])
        if val == 0:
            val = nearest_nonzero(row, col)
        if val > 0:
            if val in ins_to_room and ins_to_room[val] != rid:
                # Ambiguous: keep first assignment
                continue
            ins_to_room[val] = rid
    return ins_to_room


def rooms_near_door_from_seg(
    ins_arr: np.ndarray,
    ins_to_room: dict[int, str],
    wx: float,
    wy: float,
    img_w: int,
    img_h: int,
    resolution: float,
    exclude: set[str],
) -> set[str]:
    col, row = world_xy_to_pixel(wx, wy, img_w, img_h, resolution)
    h, w = ins_arr.shape
    found: set[str] = set()
    for dr in range(-15, 16, 3):
        for dc in range(-15, 16, 3):
            r, c = row + dr, col + dc
            if 0 <= r < h and 0 <= c < w:
                vid = int(ins_arr[r, c])
                if vid > 0 and vid in ins_to_room:
                    rn = ins_to_room[vid]
                    if rn not in exclude:
                        found.add(rn)
    return found


def pick_nearest_room(
    xy: tuple[float, float],
    candidates: set[str],
    room_xy: dict[str, list[float]],
) -> str | None:
    if not candidates:
        return None
    x0, y0 = xy
    return min(
        candidates,
        key=lambda rid: (room_xy[rid][0] - x0) ** 2 + (room_xy[rid][1] - y0) ** 2,
    )


def infer_second_room_for_door(
    _door_name: str,
    known_rooms: list[str],
    pos: list[float],
    floor_id: int,
    room_floor: dict[str, int],
    room_xy: dict[str, list[float]],
    room_ids: list[str],
    layout_dir: Path,
    scenes_root: Path,
) -> str | None:
    """Return additional room instance id for a single-room door, or None."""
    if len(known_rooms) != 1:
        return None
    k0 = known_rooms[0]
    wx, wy = float(pos[0]), float(pos[1])
    exclude = {k0}

    ins_arr, sem_arr = load_ins_sem_arrays(layout_dir, floor_id)
    if ins_arr is not None and sem_arr is not None:
        ih, iw = ins_arr.shape
        ins_to_room: dict[int, str] = {}
        rc_path = find_room_categories_path(scenes_root)
        if rc_path is not None:
            try:
                sem_map = build_sem_id_to_name(rc_path)
                ins_to_room = build_ins_id_to_room_from_sem(ins_arr, sem_arr, sem_map)
            except OSError:
                ins_to_room = {}
        if not ins_to_room:
            ins_to_room = bootstrap_ins_to_room_from_centroids(
                ins_arr, room_ids, room_xy, iw, ih, SEG_MAP_NATIVE_MPP
            )

        near = rooms_near_door_from_seg(
            ins_arr,
            ins_to_room,
            wx,
            wy,
            iw,
            ih,
            SEG_MAP_NATIVE_MPP,
            exclude,
        )
        same_floor = {r for r in near if room_floor.get(r, 0) == floor_id}
        if len(same_floor) == 1:
            return next(iter(same_floor))
        if len(same_floor) > 1:
            picked = pick_nearest_room((wx, wy), same_floor, room_xy)
            return picked

    # Proximity fallback: nearest other room on same floor
    others = [
        r
        for r in room_ids
        if r != k0 and room_floor.get(r, 0) == floor_id
    ]
    if not others:
        return None
    best = min(
        others,
        key=lambda rid: (room_xy[rid][0] - wx) ** 2 + (room_xy[rid][1] - wy) ** 2,
    )
    d = (
        (room_xy[best][0] - wx) ** 2 + (room_xy[best][1] - wy) ** 2
    ) ** 0.5
    if d <= DOOR_PROXIMITY_MAX_M:
        return best
    return None


def infer_missing_connectivity(
    G: nx.Graph,
    objects: list[dict[str, Any]],
    layout_dir: Path,
    scenes_root: Path,
    room_ids: list[str],
    room_floor: dict[str, int],
    room_xy: dict[str, list[float]],
) -> int:
    """
    Augment graph: second room for single-room doors (seg + proximity),
    then link isolated rooms to nearest neighbor (room_adjacent, inferred).
    Returns count of new edges added.
    """
    added = 0
    door_nodes = [n for n, a in G.nodes(data=True) if a.get("type") == "door"]

    for dn in door_nodes:
        attrs = G.nodes[dn]
        connected = list(attrs.get("connected_rooms") or [])
        if len(connected) != 1:
            continue
        pos = attrs.get("position") or [0.0, 0.0]
        dfloor = int(attrs.get("floor_id", 0))
        second = infer_second_room_for_door(
            dn,
            connected,
            pos,
            dfloor,
            room_floor,
            room_xy,
            room_ids,
            layout_dir,
            scenes_root,
        )
        if second is None or second not in G.nodes:
            continue
        r0 = connected[0]
        if second == r0:
            continue
        new_rooms = [r0, second]
        attrs["connected_rooms"] = new_rooms
        attrs["inferred_second_room"] = True
        if not G.has_edge(dn, second):
            G.add_edge(dn, second, relation="connected_by_door", inferred=True)
            added += 1
        if not G.has_edge(r0, second):
            G.add_edge(r0, second, relation="room_adjacent", inferred=True)
            added += 1
        # Refresh door floor if spanning floors (rare)
        floors = {room_floor.get(r, 0) for r in new_rooms}
        if len(floors) == 1:
            attrs["floor_id"] = next(iter(floors))

    # Rooms with no path to another room via door/stair edges (except contains)
    def room_has_connector(rid: str) -> bool:
        for nb in G.neighbors(rid):
            t = G.nodes[nb].get("type")
            if t in ("door", "staircase"):
                return True
        return False

    for rid in room_ids:
        if not room_has_connector(rid):
            fi = room_floor.get(rid, 0)
            others = [
                r
                for r in room_ids
                if r != rid and room_floor.get(r, 0) == fi
            ]
            if not others:
                continue
            cx, cy = room_xy[rid][0], room_xy[rid][1]
            best = min(
                others,
                key=lambda r: (room_xy[r][0] - cx) ** 2 + (room_xy[r][1] - cy) ** 2,
            )
            d = (
                (room_xy[best][0] - cx) ** 2 + (room_xy[best][1] - cy) ** 2
            ) ** 0.5
            if d <= ISOLATED_ROOM_LINK_MAX_M and not G.has_edge(rid, best):
                G.add_edge(rid, best, relation="room_adjacent", inferred=True)
                added += 1

    # Bridge disconnected room components (e.g. kitchen only linked to exterior door)
    added += bridge_disconnected_room_components(G, room_ids, room_floor, room_xy)

    return added


def bridge_disconnected_room_components(
    G: nx.Graph,
    room_ids: list[str],
    room_floor: dict[str, int],
    room_xy: dict[str, list[float]],
) -> int:
    """
    If rooms on a floor split into multiple components when connected only via
    door/stair intermediaries or direct room_adjacent, add inferred room_adjacent
    edges between nearest pairs until one component remains.
    """
    added = 0
    if not room_ids:
        return added
    max_floor = max(room_floor.get(r, 0) for r in room_ids)

    def build_room_graph(floor_rooms: set[str]) -> nx.Graph:
        rg = nx.Graph()
        rg.add_nodes_from(floor_rooms)
        for r in floor_rooms:
            for nb in G.neighbors(r):
                ntype = G.nodes[nb].get("type")
                if ntype in ("door", "staircase"):
                    for nb2 in G.neighbors(nb):
                        if nb2 != r and nb2 in floor_rooms:
                            rg.add_edge(r, nb2)
                elif ntype == "room" and nb in floor_rooms:
                    rg.add_edge(r, nb)
        return rg

    for fi in range(max_floor + 1):
        floor_rooms = {r for r in room_ids if room_floor.get(r, 0) == fi}
        if len(floor_rooms) <= 1:
            continue
        room_graph = build_room_graph(floor_rooms)
        components = list(nx.connected_components(room_graph))
        while len(components) > 1:
            best_dist = float("inf")
            best_pair: tuple[str, str] | None = None
            for i, comp_a in enumerate(components):
                for j, comp_b in enumerate(components):
                    if j <= i:
                        continue
                    for ra in comp_a:
                        for rb in comp_b:
                            d = (
                                (room_xy[ra][0] - room_xy[rb][0]) ** 2
                                + (room_xy[ra][1] - room_xy[rb][1]) ** 2
                            ) ** 0.5
                            if d < best_dist:
                                best_dist = d
                                best_pair = (ra, rb)
            if best_pair is None:
                break
            ra, rb = best_pair
            if not G.has_edge(ra, rb):
                G.add_edge(ra, rb, relation="room_adjacent", inferred=True)
                added += 1
            room_graph.add_edge(ra, rb)
            components = list(nx.connected_components(room_graph))

    return added


def trav_map_building_bbox(
    arr: np.ndarray, threshold: int = 12
) -> tuple[int, int, int, int] | None:
    """Bounding box (r0, r1, c0, c1) of non-background pixels; arr is HxWx3 or HxW."""
    if arr.ndim == 3:
        mask = np.any(arr > threshold, axis=2)
    else:
        mask = arr > threshold
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def graph_node_bbox(
    G: nx.Graph,
    floor_idx: int,
    resolution: float,
    img_w: int,
    img_h: int,
    margin_px: int = GRAPH_NODE_CROP_MARGIN_PX,
) -> tuple[int, int, int, int] | None:
    """Bounding box (r0, r1, c0, c1) from room/door/stair positions for overlay crop."""
    rows_cols: list[tuple[int, int]] = []
    for n, a in G.nodes(data=True):
        t = a.get("type")
        if t not in ("room", "door", "staircase"):
            continue
        fid = int(a.get("floor_id", 0))
        if t == "staircase":
            neighbors = list(G.neighbors(n))
            floors_touch: set[int] = set()
            for nb in neighbors:
                if nb.startswith("floor_"):
                    try:
                        floors_touch.add(int(nb.split("_")[1]))
                    except (ValueError, IndexError):
                        pass
            if floors_touch and floor_idx not in floors_touch:
                continue
        elif fid != floor_idx:
            continue
        pos = a.get("centroid")
        if pos is None:
            pos = a.get("position")
        if not pos or len(pos) < 2:
            continue
        col, row = world_xy_to_pixel(
            float(pos[0]), float(pos[1]), img_w, img_h, resolution
        )
        rows_cols.append((row, col))
    if not rows_cols:
        return None
    rows, cols = zip(*rows_cols)
    r0 = max(0, int(min(rows)) - margin_px)
    r1 = min(img_h, int(max(rows)) + margin_px)
    c0 = max(0, int(min(cols)) - margin_px)
    c1 = min(img_w, int(max(cols)) + margin_px)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, r1, c0, c1


def choose_crop_bbox(
    trav_crop: tuple[int, int, int, int],
    node_bbox: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    """Prefer tighter node-based crop when traversability bbox is much larger."""
    tr0, tr1, tc0, tc1 = trav_crop
    trav_area = max(0, tr1 - tr0) * max(0, tc1 - tc0)
    if node_bbox is None:
        return trav_crop
    nr0, nr1, nc0, nc1 = node_bbox
    node_area = max(0, nr1 - nr0) * max(0, nc1 - nc0)
    if node_area > 0 and trav_area > CROP_TRAV_VS_NODE_AREA_RATIO * node_area:
        return node_bbox
    return trav_crop


def abbreviate_label(name: str, max_len: int = 22) -> str:
    if name.startswith("door_"):
        return "d_" + name[5:]
    if len(name) <= max_len:
        return name
    return name[: max_len - 2] + ".."


def build_navigation_graph(
    scene_name: str,
    objects: list[dict[str, Any]],
    layout_dir: Path,
    scenes_root: Path | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Build NetworkX graph with typed nodes and edges."""
    G = nx.Graph()

    num_floors = count_floor_trav_maps(layout_dir)
    if num_floors == 0:
        num_floors = 1

    # All room ids from in_rooms
    room_ids_set: set[str] = set()
    for o in objects:
        room_ids_set.update(o["in_rooms"])
    room_ids = sorted(r for r in room_ids_set if r.strip())

    room_floor = assign_room_floor_ids_v2(room_ids, objects, num_floors)

    # Room centroids (xy) from floors category first
    room_xy: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    room_counts: dict[str, int] = defaultdict(int)
    for o in objects:
        if o["category"] != "floors":
            continue
        p = o["pos"]
        if p is None:
            continue
        for r in o["in_rooms"]:
            room_xy[r][0] += float(p[0])
            room_xy[r][1] += float(p[1])
            room_counts[r] += 1

    for r in room_ids:
        if room_counts[r] == 0:
            # fallback: all objects in room
            sx, sy, n = 0.0, 0.0, 0
            for o in objects:
                if r not in o["in_rooms"]:
                    continue
                p = o["pos"]
                if p is None:
                    continue
                sx += float(p[0])
                sy += float(p[1])
                n += 1
            if n > 0:
                room_xy[r] = [sx / n, sy / n]
            else:
                room_xy[r] = [0.0, 0.0]

        else:
            room_xy[r][0] /= room_counts[r]
            room_xy[r][1] /= room_counts[r]

    # --- Building (layer 0)
    G.add_node(
        scene_name,
        id=scene_name,
        type="building",
        layer=0,
    )

    # --- Floor levels (layer 1)
    for fi in range(num_floors):
        fid = f"floor_{fi}"
        G.add_node(fid, id=fid, type="floor_level", layer=1)
        G.add_edge(scene_name, fid, relation="contains")

    # --- Rooms (layer 2)
    num_doors = 0
    num_stairs = 0

    for r in room_ids:
        rt, inst = parse_room_name(r)
        is_corridor = rt in ("hallway", "corridor")
        floor_id = room_floor.get(r, 0)
        cx, cy = room_xy[r][0], room_xy[r][1]
        G.add_node(
            r,
            id=r,
            type="room",
            layer=2,
            room_type=rt,
            instance_id=inst,
            floor_id=floor_id,
            centroid=[cx, cy],
            navigable=True,
            is_corridor=is_corridor,
        )
        G.add_edge(f"floor_{floor_id}", r, relation="contains")

    # --- Doors (layer 3)
    for o in objects:
        if not is_door_category(o["category"]):
            continue
        name = o["name"]
        connected = list(o["in_rooms"])
        pos = o["pos"]
        xy = [float(pos[0]), float(pos[1])] if pos is not None else [0.0, 0.0]
        dfloor = infer_door_floor_id(connected, room_floor)
        G.add_node(
            name,
            id=name,
            type="door",
            layer=3,
            subtype=o["category"],
            floor_id=dfloor,
            position=xy,
            connected_rooms=connected,
        )
        num_doors += 1

        for room_id in connected:
            if room_id in G.nodes:
                G.add_edge(name, room_id, relation="connected_by_door")

        if len(connected) == 2:
            a, b = connected[0], connected[1]
            if a in G.nodes and b in G.nodes:
                G.add_edge(a, b, relation="room_adjacent")

    # --- Staircases: multi-floor emphasis; still add if present on single floor
    stair_objects = [o for o in objects if is_stair_category(o["category"])]
    if num_floors > 1 and not stair_objects:
        print(
            f"  [warn] {scene_name}: multi-floor scene but no staircase/stair objects found; "
            "inter-floor connectivity may be incomplete."
        )

    for o in stair_objects:
        name = o["name"]
        pos = o["pos"]
        xy = [float(pos[0]), float(pos[1])] if pos is not None else [0.0, 0.0]
        G.add_node(
            name,
            id=name,
            type="staircase",
            layer=3,
            position=xy,
        )
        num_stairs += 1

        # Nearest room per floor (by centroid xy)
        for fi in range(num_floors):
            candidates = [
                r
                for r in room_ids
                if room_floor.get(r, 0) == fi and r in room_xy
            ]
            if not candidates:
                continue
            best_r = min(
                candidates,
                key=lambda rid: (room_xy[rid][0] - xy[0]) ** 2
                + (room_xy[rid][1] - xy[1]) ** 2,
            )
            G.add_edge(name, best_r, relation="connected_by_door")

        # Connect to floor nodes
        if num_floors > 1:
            touched = sorted({room_floor.get(r, 0) for r in o["in_rooms"] if r in room_floor})
            if len(touched) >= 2:
                for fi in touched:
                    G.add_edge(name, f"floor_{fi}", relation="connects_floors")
            else:
                # nearest-room floors spanned: use min/max floor from all rooms near stair
                dists = []
                for r in room_ids:
                    d = (room_xy[r][0] - xy[0]) ** 2 + (room_xy[r][1] - xy[1]) ** 2
                    dists.append((d, room_floor.get(r, 0), r))
                dists.sort()
                floors_near = {f for _, f, _ in dists[: min(6, len(dists))]}
                for fi in floors_near:
                    G.add_edge(name, f"floor_{fi}", relation="connects_floors")
        else:
            G.add_edge(name, "floor_0", relation="connects_floors")

    inferred_edges = 0
    if scenes_root is not None:
        inferred_edges = infer_missing_connectivity(
            G,
            objects,
            layout_dir,
            scenes_root,
            room_ids,
            room_floor,
            room_xy,
        )

    meta = {
        "num_floors": num_floors,
        "num_rooms": len(room_ids),
        "num_doors": num_doors,
        "num_staircases": num_stairs,
        "inferred_connections": inferred_edges,
    }
    return G, meta


def node_color_and_size(attrs: dict[str, Any]) -> tuple[str, int]:
    t = attrs.get("type", "")
    if t == "building":
        return "#2F4F4F", 1400
    if t == "floor_level":
        return "#888888", 1200
    if t == "room":
        if attrs.get("is_corridor"):
            return "#9370DB", 900
        return "#4682B4", 900
    if t == "door":
        return "#FF8C00", 500
    if t == "staircase":
        return "#32CD32", 700
    return "#AAAAAA", 400


def draw_scene_graph_png(
    G: nx.Graph, scene_name: str, out_path: Path, dpi: int = 150
) -> None:
    plt.figure(figsize=(14, 10), dpi=dpi)
    pos = None
    try:
        pos = nx.kamada_kawai_layout(G, weight=None)
    except Exception:
        try:
            pos = nx.spring_layout(G, seed=42)
        except Exception:
            pos = nx.circular_layout(G)

    node_colors = []
    node_sizes = []
    for n in G.nodes():
        c, s = node_color_and_size(G.nodes[n])
        node_colors.append(c)
        node_sizes.append(s)

    # Draw edges by relation
    contains_e = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "contains"]
    door_e = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "connected_by_door"
    ]
    adj_e = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "room_adjacent"
    ]
    stair_e = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "connects_floors"
    ]
    other_e = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("relation") not in ("contains", "connected_by_door", "room_adjacent", "connects_floors")
    ]

    nx.draw_networkx_edges(G, pos, edgelist=contains_e, edge_color="#D3D3D3", width=1.2, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=door_e, edge_color="#FF8C00", width=1.5, alpha=0.9)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=adj_e,
        edge_color="#4682B4",
        width=1.2,
        style="dashed",
        alpha=0.8,
    )
    nx.draw_networkx_edges(G, pos, edgelist=stair_e, edge_color="#228B22", width=2.0, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=other_e, edge_color="#CCCCCC", width=1.0, alpha=0.5)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"{scene_name}  |  navigation scene graph", fontsize=12)
    # Legend
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor="#2F4F4F", edgecolor="k", label="building"),
        Patch(facecolor="#888888", edgecolor="k", label="floor_level"),
        Patch(facecolor="#4682B4", edgecolor="k", label="room"),
        Patch(facecolor="#9370DB", edgecolor="k", label="hallway/corridor"),
        Patch(facecolor="#FF8C00", edgecolor="k", label="door"),
        Patch(facecolor="#32CD32", edgecolor="k", label="staircase"),
    ]
    plt.legend(handles=legend_elems, loc="upper left", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _trav_png_sort_key(p: Path) -> int:
    m = _FLOOR_TRAV_RE.match(p.name)
    return int(m.group(1)) if m else 0


def draw_overlays(
    G: nx.Graph,
    layout_dir: Path,
    scene_name: str,
    out_dir: Path,
    resolution: float,
) -> None:
    """One overlay PNG per floor trav map; auto-cropped footprint, scaled fonts, short door labels."""
    from matplotlib.patches import Circle, Polygon, Rectangle

    if not layout_dir.is_dir():
        print(f"  [warn] {scene_name}: no layout dir, skipping overlay")
        return

    trav_paths = sorted(layout_dir.glob("floor_trav_*.png"), key=_trav_png_sort_key)
    if not trav_paths:
        print(f"  [warn] {scene_name}: no floor_trav_*.png in layout, skipping overlay")
        return

    label_bbox = {"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.78}

    for tp in trav_paths:
        m = _FLOOR_TRAV_RE.match(tp.name)
        if not m:
            continue
        floor_idx = int(m.group(1))
        try:
            img = Image.open(tp).convert("RGB")
        except OSError as e:
            print(f"  [warn] could not open {tp}: {e}")
            continue

        w, h = img.size
        arr = np.asarray(img)
        bbox = trav_map_building_bbox(arr)
        margin = OVERLAY_CROP_MARGIN_PX
        if bbox is not None:
            r0, r1, c0, c1 = bbox
            r0 = max(0, r0 - margin)
            r1 = min(h, r1 + margin)
            c0 = max(0, c0 - margin)
            c1 = min(w, c1 + margin)
        else:
            r0, r1, c0, c1 = 0, h, 0, w

        trav_crop = (r0, r1, c0, c1)
        node_bbox = graph_node_bbox(G, floor_idx, resolution, w, h)
        r0, r1, c0, c1 = choose_crop_bbox(trav_crop, node_bbox)

        arr_crop = arr[r0:r1, c0:c1]
        cw, ch = c1 - c0, r1 - r0
        base = max(cw, ch, 1)
        fs = max(5.5, min(10.0, 900.0 / base))
        r_pix = max(6, min(14, int(base / 55)))

        def to_px(wx: float, wy: float) -> tuple[float, float]:
            col, row = world_xy_to_pixel(wx, wy, w, h, resolution)
            return float(col - c0), float(row - r0)

        def node_floor(n: str) -> int | None:
            if n not in G.nodes:
                return None
            a = G.nodes[n]
            if a.get("type") == "room":
                return int(a.get("floor_id", 0))
            if a.get("type") == "door":
                return int(a.get("floor_id", 0))
            if a.get("type") == "staircase":
                return floor_idx
            return None

        dpi = 150
        fig = plt.figure(figsize=(cw / dpi, ch / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr_crop, origin="upper", aspect="equal")
        ax.set_axis_off()

        # Edges
        for u, v, d in G.edges(data=True):
            rel = d.get("relation")
            if rel == "room_adjacent":
                fu = node_floor(u)
                fv = node_floor(v)
                if fu == floor_idx and fv == floor_idx:
                    if u in G.nodes and "centroid" in G.nodes[u]:
                        c1 = G.nodes[u]["centroid"]
                        p1 = to_px(float(c1[0]), float(c1[1]))
                    else:
                        continue
                    if v in G.nodes and "centroid" in G.nodes[v]:
                        c2 = G.nodes[v]["centroid"]
                        p2 = to_px(float(c2[0]), float(c2[1]))
                    else:
                        continue
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        color=(70 / 255, 130 / 255, 180 / 255, 0.59),
                        linewidth=2,
                        solid_capstyle="round",
                    )

            if rel == "connected_by_door":
                pts = []
                for n in (u, v):
                    if n not in G.nodes:
                        continue
                    a = G.nodes[n]
                    if a.get("type") == "room" and int(a.get("floor_id", 0)) == floor_idx:
                        c = a.get("centroid")
                        if c:
                            pts.append(to_px(float(c[0]), float(c[1])))
                    elif a.get("type") == "door" and int(a.get("floor_id", 0)) == floor_idx:
                        p = a.get("position")
                        if p:
                            pts.append(to_px(float(p[0]), float(p[1])))
                if len(pts) == 2:
                    ax.plot(
                        [pts[0][0], pts[1][0]],
                        [pts[0][1], pts[1][1]],
                        color=(1.0, 140 / 255, 0.0, 0.59),
                        linewidth=2,
                        solid_capstyle="round",
                    )

        # Rooms (circles)
        for n, a in G.nodes(data=True):
            if a.get("type") != "room":
                continue
            if int(a.get("floor_id", 0)) != floor_idx:
                continue
            c = a.get("centroid")
            if not c:
                continue
            px, py = to_px(float(c[0]), float(c[1]))
            face = (
                (147 / 255, 112 / 255, 219 / 255, 0.78)
                if a.get("is_corridor")
                else (70 / 255, 130 / 255, 180 / 255, 0.78)
            )
            ax.add_patch(
                Circle(
                    (px, py),
                    r_pix,
                    facecolor=face,
                    edgecolor="black",
                    linewidth=1,
                )
            )

        # Doors (squares)
        for n, a in G.nodes(data=True):
            if a.get("type") != "door":
                continue
            if int(a.get("floor_id", 0)) != floor_idx:
                continue
            p = a.get("position")
            if not p:
                continue
            px, py = to_px(float(p[0]), float(p[1]))
            s = max(5, min(10, r_pix - 2))
            ax.add_patch(
                Rectangle(
                    (px - s, py - s),
                    2 * s,
                    2 * s,
                    facecolor=(1.0, 140 / 255, 0.0, 0.86),
                    edgecolor="black",
                    linewidth=1,
                )
            )

        # Staircases (triangles)
        for n, a in G.nodes(data=True):
            if a.get("type") != "staircase":
                continue
            p = a.get("position")
            if not p:
                continue
            px, py = to_px(float(p[0]), float(p[1]))
            neighbors = list(G.neighbors(n))
            floors_touch = set()
            for nb in neighbors:
                if nb.startswith("floor_"):
                    try:
                        floors_touch.add(int(nb.split("_")[1]))
                    except (ValueError, IndexError):
                        pass
            if not floors_touch or floor_idx in floors_touch:
                tri = Polygon(
                    [(px, py - 10), (px - 9, py + 8), (px + 9, py + 8)],
                    closed=True,
                    facecolor=(50 / 255, 205 / 255, 50 / 255, 0.86),
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(tri)

        # Labels: stagger offsets in points to reduce overlap
        label_positions: list[tuple[float, float]] = []
        li = 0
        for n, a in G.nodes(data=True):
            label = None
            xyw = None
            if a.get("type") == "room" and int(a.get("floor_id", 0)) == floor_idx:
                c = a.get("centroid")
                if c:
                    xyw = to_px(float(c[0]), float(c[1]))
                    label = str(n)
            elif a.get("type") == "door" and int(a.get("floor_id", 0)) == floor_idx:
                p = a.get("position")
                if p:
                    xyw = to_px(float(p[0]), float(p[1]))
                    label = abbreviate_label(str(n))
            elif a.get("type") == "staircase":
                p = a.get("position")
                if p:
                    neighbors = list(G.neighbors(n))
                    floors_touch = set()
                    for nb in neighbors:
                        if nb.startswith("floor_"):
                            try:
                                floors_touch.add(int(nb.split("_")[1]))
                            except (ValueError, IndexError):
                                pass
                    if not floors_touch or floor_idx in floors_touch:
                        xyw = to_px(float(p[0]), float(p[1]))
                        label = abbreviate_label(str(n))
            if label and xyw:
                ox = 4 + (li % 4) * 5
                oy = -4 - (li % 3) * 6
                for ex, ey in label_positions:
                    if abs(xyw[0] - ex) < 3.0 and abs(xyw[1] - ey) < 3.0:
                        oy -= 18
                        break
                label_positions.append((xyw[0], xyw[1]))
                li += 1
                ax.annotate(
                    label,
                    xy=(xyw[0], xyw[1]),
                    xytext=(ox, oy),
                    textcoords="offset points",
                    fontsize=fs,
                    color="black",
                    bbox=label_bbox,
                    ha="left",
                    va="bottom",
                )

        if len(trav_paths) == 1:
            out_img = out_dir / "scene_graph_overlay.png"
        else:
            out_img = out_dir / f"scene_graph_overlay_floor_{floor_idx}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_img, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  wrote {out_img}")


def draw_birds_eye_layout(
    G: nx.Graph,
    layout_dir: Path,
    scene_name: str,
    out_dir: Path,
    resolution: float,
    scenes_root: Path,
) -> None:
    """
    Clean 2D bird's-eye layout: traversability (cropped), optional room segmentation tint,
    room labels, door/stair markers, connectivity between room centroids (no door labels).
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle, Patch, Polygon, Rectangle

    if not layout_dir.is_dir():
        return

    trav_paths = sorted(layout_dir.glob("floor_trav_*.png"), key=_trav_png_sort_key)
    if not trav_paths:
        return

    # Stable color per room id
    def room_color(rid: str) -> tuple[float, float, float, float]:
        hue = (hash(rid) % 1000) / 1000.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.45, 0.92)
        return (r, g, b, 0.42)

    for tp in trav_paths:
        m = _FLOOR_TRAV_RE.match(tp.name)
        if not m:
            continue
        floor_idx = int(m.group(1))
        try:
            img = Image.open(tp).convert("RGB")
        except OSError:
            continue

        w, h = img.size
        arr_u8 = np.asarray(img)
        bbox = trav_map_building_bbox(arr_u8)
        arr = arr_u8.astype(np.float32) / 255.0
        margin = OVERLAY_CROP_MARGIN_PX
        if bbox is not None:
            r0, r1, c0, c1 = bbox
            r0 = max(0, r0 - margin)
            r1 = min(h, r1 + margin)
            c0 = max(0, c0 - margin)
            c1 = min(w, c1 + margin)
        else:
            r0, r1, c0, c1 = 0, h, 0, w

        trav_crop = (r0, r1, c0, c1)
        node_bbox = graph_node_bbox(G, floor_idx, resolution, w, h)
        r0, r1, c0, c1 = choose_crop_bbox(trav_crop, node_bbox)

        arr_crop = arr[r0:r1, c0:c1]
        cw, ch = c1 - c0, r1 - r0

        room_ids = sorted(
            n for n, a in G.nodes(data=True) if a.get("type") == "room"
        )
        room_floor = {n: int(G.nodes[n].get("floor_id", 0)) for n in room_ids}

        def to_px(wx: float, wy: float) -> tuple[float, float]:
            col, row = world_xy_to_pixel(wx, wy, w, h, resolution)
            return float(col - c0), float(row - r0)

        dpi = 150
        fig = plt.figure(figsize=(cw / dpi, ch / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr_crop, origin="upper", aspect="equal")

        ins_arr, sem_arr = load_ins_sem_arrays(layout_dir, floor_idx)
        if (
            ins_arr is not None
            and ins_arr.shape[0] == h
            and ins_arr.shape[1] == w
        ):
            ins_crop = ins_arr[r0:r1, c0:c1]
            room_xy_map: dict[str, list[float]] = {}
            for n, a in G.nodes(data=True):
                if a.get("type") == "room" and "centroid" in a:
                    room_xy_map[n] = a["centroid"]
            ih, iw = ins_arr.shape
            ins_to_room: dict[int, str] = {}
            rc_path = find_room_categories_path(scenes_root)
            if rc_path is not None and sem_arr is not None:
                try:
                    sem_map = build_sem_id_to_name(rc_path)
                    ins_to_room = build_ins_id_to_room_from_sem(ins_arr, sem_arr, sem_map)
                except OSError:
                    ins_to_room = {}
            if not ins_to_room:
                ins_to_room = bootstrap_ins_to_room_from_centroids(
                    ins_arr, room_ids, room_xy_map, iw, ih, SEG_MAP_NATIVE_MPP
                )

            overlay = np.zeros((ch, cw, 4), dtype=np.float32)
            for ins_id, rn in ins_to_room.items():
                if ins_id == 0 or rn not in room_ids:
                    continue
                if room_floor.get(rn, 0) != floor_idx:
                    continue
                mask = ins_crop == ins_id
                if not np.any(mask):
                    continue
                rgba = room_color(rn)
                overlay[mask] = rgba
            ax.imshow(overlay, origin="upper", aspect="equal")

        # Room–room connectivity (centroid to centroid)
        drawn: set[tuple[str, str]] = set()

        def draw_rr(a: str, b: str) -> None:
            if a > b:
                a, b = b, a
            if (a, b) in drawn:
                return
            drawn.add((a, b))
            na, nb = G.nodes[a], G.nodes[b]
            if na.get("type") != "room" or nb.get("type") != "room":
                return
            if int(na.get("floor_id", 0)) != floor_idx or int(nb.get("floor_id", 0)) != floor_idx:
                return
            ca, cb = na.get("centroid"), nb.get("centroid")
            if not ca or not cb:
                return
            p1 = to_px(float(ca[0]), float(ca[1]))
            p2 = to_px(float(cb[0]), float(cb[1]))
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=(0.15, 0.15, 0.2, 0.55),
                linewidth=1.2,
                solid_capstyle="round",
            )

        for u, v, d in G.edges(data=True):
            if d.get("relation") == "room_adjacent":
                draw_rr(u, v)
        for _dn, dattrs in G.nodes(data=True):
            if dattrs.get("type") != "door":
                continue
            if int(dattrs.get("floor_id", 0)) != floor_idx:
                continue
            cr = dattrs.get("connected_rooms") or []
            if len(cr) >= 2 and cr[0] in G.nodes and cr[1] in G.nodes:
                draw_rr(cr[0], cr[1])

        fs = max(6.0, min(11.0, 950.0 / max(cw, ch, 1)))
        r_dot = max(5, min(12, int(max(cw, ch) / 70)))

        room_centroids_px: list[tuple[float, float]] = []
        for n, a in G.nodes(data=True):
            if a.get("type") != "room":
                continue
            if int(a.get("floor_id", 0)) != floor_idx:
                continue
            c = a.get("centroid")
            if c:
                room_centroids_px.append(to_px(float(c[0]), float(c[1])))

        room_label_positions: list[tuple[float, float]] = []
        for n, a in G.nodes(data=True):
            if a.get("type") != "room":
                continue
            if int(a.get("floor_id", 0)) != floor_idx:
                continue
            c = a.get("centroid")
            if not c:
                continue
            px, py = to_px(float(c[0]), float(c[1]))
            ax.add_patch(
                Circle(
                    (px, py),
                    r_dot,
                    facecolor=to_rgba("#4682B4", 0.55),
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
            ox_pt, oy_pt = 0, 2
            for ex, ey in room_label_positions:
                if abs(px - ex) < 3.0 and abs(py - ey) < 3.0:
                    oy_pt += 14
                    break
            room_label_positions.append((px, py))
            ax.annotate(
                str(n),
                xy=(px, py),
                xytext=(ox_pt, oy_pt),
                textcoords="offset points",
                fontsize=fs,
                ha="center",
                va="bottom",
                color="black",
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "alpha": 0.82},
            )

        for n, a in G.nodes(data=True):
            if a.get("type") != "door":
                continue
            if int(a.get("floor_id", 0)) != floor_idx:
                continue
            p = a.get("position")
            if not p:
                continue
            px, py = to_px(float(p[0]), float(p[1]))
            for rcx, rcy in room_centroids_px:
                if abs(px - rcx) < 3.0 and abs(py - rcy) < 3.0:
                    px += 10.0
                    py += 10.0
                    break
            s = max(3, r_dot // 2)
            ax.add_patch(
                Rectangle(
                    (px - s, py - s),
                    2 * s,
                    2 * s,
                    facecolor="#FF8C00",
                    edgecolor="black",
                    linewidth=0.6,
                )
            )

        for n, a in G.nodes(data=True):
            if a.get("type") != "staircase":
                continue
            p = a.get("position")
            if not p:
                continue
            px, py = to_px(float(p[0]), float(p[1]))
            neighbors = list(G.neighbors(n))
            floors_touch = set()
            for nb in neighbors:
                if nb.startswith("floor_"):
                    try:
                        floors_touch.add(int(nb.split("_")[1]))
                    except (ValueError, IndexError):
                        pass
            if not floors_touch or floor_idx in floors_touch:
                ax.add_patch(
                    Polygon(
                        [(px, py - 8), (px - 7, py + 6), (px + 7, py + 6)],
                        closed=True,
                        facecolor="#32CD32",
                        edgecolor="black",
                        linewidth=0.7,
                    )
                )

        legend_elems = [
            Patch(facecolor="#4682B4", edgecolor="k", label="room", alpha=0.55),
            Patch(facecolor="#FF8C00", edgecolor="k", label="door"),
            Patch(facecolor="#32CD32", edgecolor="k", label="staircase"),
        ]
        ax.legend(handles=legend_elems, loc="upper right", fontsize=max(5, fs - 2))
        ax.set_title(f"{scene_name}  |  bird's-eye layout (floor {floor_idx})", fontsize=fs + 1)
        ax.set_axis_off()

        if len(trav_paths) == 1:
            out_img = out_dir / "birds_eye_layout.png"
        else:
            out_img = out_dir / f"birds_eye_layout_floor_{floor_idx}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_img, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        print(f"  wrote {out_img}")


def graph_to_json_envelope(
    G: nx.Graph, scene_name: str, meta: dict[str, Any]
) -> dict[str, Any]:
    # edges="links" keeps NetworkX 3.6+ from changing the output key name.
    data = nx.node_link_data(G, edges="links")
    return {
        "scene": scene_name,
        "generator": "build_scene_graphs.py",
        "design_reference": "ConceptGraphs (arXiv:2309.16650) hierarchical scene graph",
        "hierarchy_layers": ["building", "floor_level", "room", "connector"],
        "edge_relations": [
            "contains",
            "connected_by_door",
            "room_adjacent",
            "connects_floors",
        ],
        "num_floors": meta["num_floors"],
        "num_rooms": meta["num_rooms"],
        "num_doors": meta["num_doors"],
        "num_staircases": meta["num_staircases"],
        "inferred_connections": meta.get("inferred_connections", 0),
        "graph": data,
    }


def write_readme(path: Path) -> None:
    text = """SCENE GRAPHS — Navigation structural graphs for BEHAVIOR-1K
================================================================

HOW GRAPHS WERE GENERATED
-------------------------
- Parsed scene JSON files (*_best.json) under behavior-1k-assets/scenes/<scene>/json/.
- No simulator: only objects_info.init_info and state.registry.object_registry (or legacy state.object_registry).
- Built with NetworkX; exported as node_link_data JSON.

DESIGN (ConceptGraphs-inspired)
--------------------------------
- Hierarchical layers: building -> floor_level -> room -> connector (door/staircase).
- Nodes carry typed attributes (position/centroid, room_type, floor_id, etc.).
- Edges carry a \"relation\" field for navigation semantics.

INFERRED CONNECTIVITY
---------------------
- Doors with only one room in JSON \"in_rooms\" are augmented using layout/floor_insseg_*.png
  and floor_semseg_*.png (same convention as OmniGibson SegmentationMap), plus optional
  behavior-1k-assets/metadata/room_categories.txt for semantic IDs.
- If segmentation is missing, a proximity heuristic links the door to the nearest other room
  on the same floor (within a few meters).
- Rooms still without any door/stair neighbor get an optional room_adjacent link to the nearest
  same-floor room (inferred), so the navigation subgraph is less fragmented.
- If rooms still form multiple disconnected components (e.g. a kitchen linked only to an exterior door),
  additional inferred room_adjacent edges join the nearest pair of rooms across components until
  the per-floor room graph is connected.
- Edges and door nodes may carry inferred=True; door nodes may set inferred_second_room=True.
- scene_graph.json includes \"inferred_connections\": count of inferred edges added.

OUTPUT IMAGES
-------------
- scene_graph.png — abstract graph layout (Kamada–Kawai / spring).
- scene_graph_overlay(_floor_N).png — traversability map with graph overlaid (auto-cropped to
  building footprint or, when the trav map includes a huge exterior margin, to the graph node
  extent; scaled fonts, shortened door labels).
- birds_eye_layout(_floor_N).png — 2D bird's-eye view: cropped trav map, optional room
  segmentation tint, room labels, door/stair markers, room–room connectivity (no door names).

NODE TYPES
----------
- building: root node; id is the scene name.
- floor_level: id floor_0, floor_1, ...
- room: id matches BEHAVIOR room instance (e.g. bedroom_0). Attributes include room_type, instance_id, floor_id, centroid [x,y], navigable, is_corridor for hallway/corridor.
- door: instance id from dataset; subtype is category string; position [x,y]; connected_rooms (may include an inferred second room).
- staircase: instance id; position [x,y].

EDGE RELATIONS
--------------
- contains: building-floor_level, floor_level-room.
- connected_by_door: door-room; staircase-room (nearest room per floor heuristic).
- room_adjacent: room-room when a door connects two rooms, or inferred adjacency.
- connects_floors: staircase-floor_level (inter-floor linkage).

COORDINATE SYSTEM
-----------------
- World X,Y in meters; Z ignored for 2D graph layout and overlay (except floor_id heuristics use mean Z).
- Traversability overlay uses origin at image center: pixel = world/resolution + image_size/2 (see OmniGibson BaseMap; default resolution 0.01 m/pixel for source PNGs).
- Instance segmentation maps use 0.01 m/pixel native resolution (OmniGibson default) when sampling room IDs at door positions.

KNOWN LIMITATIONS
-----------------
- Room centroids are estimated from \"floors\" object positions, not true room polygons.
- Inferred links are heuristics, not guaranteed to match true architectural topology.
- Staircases may be absent; inter-floor links may be incomplete; house_double_floor_lower/upper are separate scenes.
- floor_id for multi-floor scenes is best-effort (layout floor_trav_* count + mean-Z binning).
- Overlay world→pixel uses meters-per-pixel from scene JSON (trav_map_resolution, resolution, map_resolution, seg_map_resolution in json/*.json) when present; otherwise uses --overlay-mpp (default 0.01 m/px, OmniGibson trav PNG native). If the overlay looks misaligned, try --overlay-mpp 0.1.

"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def process_scene(
    scene_name: str,
    scenes_root: Path,
    output_root: Path,
    overlay_mpp: float,
) -> tuple[str, dict[str, Any]]:
    """Returns (status, stats_or_error)."""
    scene_dir = scenes_root / scene_name
    json_path = scene_dir / "json" / f"{scene_name}_best.json"
    layout_dir = scene_dir / "layout"
    out_dir = output_root / scene_name

    if not json_path.is_file():
        return "ERROR: missing JSON", {"error": str(json_path)}

    objects = load_scene_objects(json_path)
    if not objects:
        return "ERROR: no objects parsed", {}

    G, meta = build_navigation_graph(scene_name, objects, layout_dir, scenes_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    envelope = graph_to_json_envelope(G, scene_name, meta)
    with open(out_dir / "scene_graph.json", "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2)

    draw_scene_graph_png(G, scene_name, out_dir / "scene_graph.png")

    # Overlay: prefer metadata JSON in scene folder; else CLI overlay_mpp (default MAP_DEFAULT_RESOLUTION).
    res = find_resolution_m_per_pixel(scene_dir)
    if res is None:
        res = overlay_mpp
        print(
            f"  [warn] {scene_name}: no trav_map_resolution / map_resolution in scene JSON; "
            f"using overlay resolution {res} m/pixel. "
            f"If the overlay looks misaligned, try --overlay-mpp 0.1 or 0.01."
        )
    else:
        print(f"  [info] {scene_name}: overlay resolution from scene metadata: {res} m/pixel")

    try:
        draw_overlays(G, layout_dir, scene_name, out_dir, res)
    except Exception as e:
        print(f"  [warn] {scene_name}: overlay failed: {e}")
        traceback.print_exc()

    try:
        draw_birds_eye_layout(G, layout_dir, scene_name, out_dir, res, scenes_root)
    except Exception as e:
        print(f"  [warn] {scene_name}: bird's-eye layout failed: {e}")
        traceback.print_exc()

    stats = {
        "floors": meta["num_floors"],
        "rooms": meta["num_rooms"],
        "doors": meta["num_doors"],
        "stairs": meta["num_staircases"],
        "inferred": meta.get("inferred_connections", 0),
    }
    return "OK", stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build navigation scene graphs from BEHAVIOR-1K scene JSONs."
    )
    parser.add_argument(
        "--scenes-root",
        type=Path,
        default=None,
        help="Directory containing scene folders (default: BEHAVIOR-1K/datasets/behavior-1k-assets/scenes from repo root).",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=Path("selected_scenes.txt"),
        help="Text file with one scene name per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/scene_graphs"),
        help="Output directory for graphs and images.",
    )
    parser.add_argument(
        "--overlay-mpp",
        type=float,
        default=MAP_DEFAULT_RESOLUTION,
        metavar="METERS",
        help=(
            "Meters per pixel for overlay when scene JSON has no resolution "
            f"(default: {MAP_DEFAULT_RESOLUTION}, OmniGibson trav PNG native). "
            "Use 0.1 if your assets use a coarser map scale."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.scenes_root is None:
        env = os.environ.get("OMNIGIBSON_DATA_PATH")
        if env:
            data = Path(os.path.expanduser(env))
        else:
            data = script_dir / "BEHAVIOR-1K" / "datasets"
        args.scenes_root = data / "behavior-1k-assets" / "scenes"

    scenes_root = args.scenes_root.resolve()
    output_dir = args.output_dir.resolve()

    if args.scene_list.is_file():
        lines = [
            ln.strip()
            for ln in args.scene_list.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        scene_names = lines if lines else list(DEFAULT_SCENE_LIST)
    else:
        print(f"Scene list not found at {args.scene_list}, using hardcoded default list.")
        scene_names = list(DEFAULT_SCENE_LIST)

    # Print one parsed-object sample from the first scene that has a valid JSON (plan: verify key paths).
    for sn in scene_names:
        jp = scenes_root / sn / "json" / f"{sn}_best.json"
        if not jp.is_file():
            continue
        objs = load_scene_objects(jp)
        if objs:
            print("\n--- First object schema sample (verify key paths) ---")
            print(json.dumps(objs[0], indent=2))
            print("--- end sample ---\n")
            break

    summary_rows: list[tuple[str, int, int, int, int, int, str]] = []

    for scene_name in scene_names:
        json_path = scenes_root / scene_name / "json" / f"{scene_name}_best.json"
        try:
            status, info = process_scene(
                scene_name, scenes_root, output_dir, args.overlay_mpp
            )
            if status == "OK":
                summary_rows.append(
                    (
                        scene_name,
                        info["floors"],
                        info["rooms"],
                        info["doors"],
                        info["stairs"],
                        info.get("inferred", 0),
                        status,
                    )
                )
            else:
                err = info.get("error", "")
                summary_rows.append(
                    (scene_name, 0, 0, 0, 0, 0, f"{status} {err}".strip())
                )
        except Exception as e:
            print(f"ERROR processing {scene_name}: {e}")
            traceback.print_exc()
            summary_rows.append((scene_name, 0, 0, 0, 0, 0, f"ERROR: {e}"))

    write_readme(output_dir / "SCENE_GRAPHS_README.txt")

    # Summary table
    col0 = max(len(r[0]) for r in summary_rows) if summary_rows else 12
    h = (
        f"{'scene_name'.ljust(col0)} | floors | rooms | doors | stairs | inferred | status"
    )
    sep = "-" * len(h)
    print("\n" + h)
    print(sep)
    for name, fl, rm, dr, st, inf, stat in summary_rows:
        print(
            f"{name.ljust(col0)} | {fl:^6} | {rm:^5} | {dr:^5} | {st:^6} | {inf:^8} | {stat}"
        )


if __name__ == "__main__":
    main()
