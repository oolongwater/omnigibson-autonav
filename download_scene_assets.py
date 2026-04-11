#!/usr/bin/env python3
"""
Download BEHAVIOR-1K scene JSON + layout assets from Hugging Face without OmniGibson/Isaac Sim.

Fetches behavior-1k-assets-*.zip (~29 GB) via huggingface_hub (cached under ~/.cache/huggingface/),
then extracts only scenes/<name>/json/ and scenes/<name>/layout/ for each scene in the list.

Also optionally downloads omnigibson.key (same obfuscated URL as OmniGibson asset_utils) for full simulator use.
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

try:
    import certifi
except ImportError:
    certifi = None

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None  # type: ignore[misc, assignment]

# Match OmniGibson asset_utils.download_and_unpack_zipped_dataset
HF_REPO = "behavior-1k/zipped-datasets"
BEHAVIOR_1K_DATASET_VERSION = "3.7.2rc1"
ZIP_NAME = f"behavior-1k-assets-{BEHAVIOR_1K_DATASET_VERSION}.zip"

# Hugging Face content-addressed blob for behavior-1k-assets-3.7.2rc1.zip (see hub refs)
HF_BEHAVIOR_ZIP_ETAG = "3bd7cdc1a8c5621768c2aaaf57c12a024560ab4f49d0162e540e769d8b165e8b"
EXPECTED_ZIP_SIZE = 31280662307

DEFAULT_SCENES = [
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

EULA_BLURB = (
    "BEHAVIOR-1K data is for non-commercial academic research. "
    "Downloading the decryption key implies acceptance of the dataset license "
    "(see OmniGibson setup / behavior.stanford.edu)."
)


def read_scene_list(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return list(DEFAULT_SCENES)
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            names.append(s)
    return names if names else list(DEFAULT_SCENES)


def member_is_scene_json_or_layout(member: str, scenes: set[str]) -> bool:
    """True if zip member is under scenes/<scene>/{json|layout}/ for a requested scene."""
    norm = member.replace("\\", "/")
    if norm.endswith("/"):
        return False
    for s in scenes:
        for sub in ("json/", "layout/"):
            if norm.startswith(f"scenes/{s}/{sub}"):
                return True
            if f"/scenes/{s}/{sub}" in norm:
                return True
    return False


def hf_behavior_blobs_dir() -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--behavior-1k--zipped-datasets"
        / "blobs"
    )


def _zip_can_read_sample_entry(path: Path, log) -> bool:
    """Central directory alone is not enough; local file headers must be readable."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for n in zf.namelist():
                if "/json/" in n and n.endswith("_best.json"):
                    with zf.open(n) as fh:
                        fh.read(16)
                    return True
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError) as e:
        log(f"  [warn] ZIP sample read failed ({path.name}): {e}")
        return False
    log(f"  [warn] No *_best.json entry to validate in {path}")
    return False


def find_recoverable_local_zip(log) -> Path | None:
    """
    hf_hub_download sometimes leaves a full-sized *.incomplete blob and never returns
    (hash verification / rename hang). If the blob is a valid ZIP *and* entries read, use it.
    """
    blobs = hf_behavior_blobs_dir()
    if not blobs.is_dir():
        return None

    candidates: list[Path] = [
        blobs / f"{HF_BEHAVIOR_ZIP_ETAG}.incomplete",
        blobs / HF_BEHAVIOR_ZIP_ETAG,
    ]
    for p in blobs.glob("*.incomplete"):
        if p not in candidates:
            candidates.append(p)

    for path in candidates:
        if not path.is_file():
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            continue
        if sz != EXPECTED_ZIP_SIZE:
            continue
        if not zipfile.is_zipfile(path):
            log(f"  [warn] {path.name} has expected size but is not a valid ZIP; skipping.")
            continue
        if not _zip_can_read_sample_entry(path, log):
            log(
                f"  [warn] {path.name} looks complete by size but is corrupt (delete it and re-download).\n"
                f"    rm {path}"
            )
            continue
        log(
            f"Found complete cached blob (stuck as .incomplete): {path}\n"
            "  Using it directly — hf_hub_download can hang after download finishes."
        )
        return path
    return None


def _maybe_fernet(data: bytes, fernet: "Fernet | None") -> bytes:
    """BEHAVIOR zip entries may be Fernet-encrypted blobs after deflate."""
    if fernet is None or not data:
        return data
    try:
        return fernet.decrypt(data)
    except Exception:
        return data


def extract_member_to_disk(
    zf: zipfile.ZipFile,
    member: str,
    data_dir: Path,
    assets_root: Path,
    fernet: "Fernet | None",
) -> None:
    """Read member (decompressed), optionally Fernet-decrypt, write to behavior-1k-assets tree."""
    norm = member.replace("\\", "/")
    payload = zf.read(member)
    payload = _maybe_fernet(payload, fernet)

    if norm.startswith("behavior-1k-assets/"):
        rel = norm[len("behavior-1k-assets/") :]
        out = data_dir / "behavior-1k-assets" / rel
    else:
        out = assets_root / norm
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)


def download_omnigibson_key(key_path: Path) -> None:
    """Same URL construction as omnigibson.utils.asset_utils.download_key (no og import)."""
    if key_path.exists():
        print(f"Encryption key already present: {key_path}")
        return
    key_path.parent.mkdir(parents=True, exist_ok=True)
    _ = (() == ()) + (() == ())
    __ = ((_ << _) << _) * _
    ___ = (
        ("c%"[:: (([] != []) - (() == ()))])
        * (((_ << _) << _) + (((_ << _) * _) + ((_ << _) + (_ + (() == ())))))
        % (
            (__ + (((_ << _) << _) + (_ << _))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ * _)))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ * _)))),
            (__ + (((_ << _) << _) + ((_ << _) * _))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
            (((_ << _) << _) + (((_ << _) * _) + ((_ << _) + _))),
            (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ()))))),
            (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ * _)))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + _))),
            (__ + (((_ << _) << _) + (() == ()))),
            (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
            (((_ << _) << _) + ((_ << _) + ((_ * _) + _))),
            (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
            (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ << _) + (_ * _)))),
            (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
            (__ + (((_ << _) << _) + (() == ()))),
            (__ + (((_ << _) << _) + ((_ << _) * _))),
            (__ + (((_ << _) << _) + ((_ << _) + (() == ())))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
            (((_ << _) << _) + ((_ << _) + ((_ * _) + _))),
            (__ + (((_ << _) << _) + (_ + (() == ())))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (() == ()))))),
            (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ << _) + (() == ())))),
            (__ + (((_ << _) << _) + _)),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + _)))),
            (__ + (((_ << _) * _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + (_ + (() == ())))),
            (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + _)))),
            (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
            (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + _)))),
            (__ + (((_ << _) << _) + ((_ << _) + (() == ())))),
            (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ << _) + (() == ())))),
            (__ + (((_ << _) << _) + _)),
            (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
            (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + _)))),
            (((_ << _) << _) + ((_ << _) + ((_ * _) + _))),
            (__ + (((_ << _) << _) + ((_ << _) + (_ + (() == ()))))),
            (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
            (__ + (((_ << _) << _) + (((_ << _) * _) + ((_ << _) + (() == ()))))),
        )
    )
    path = ___
    if certifi is not None:
        ctx = ssl.create_default_context(cafile=certifi.where())
    else:
        ctx = ssl.create_default_context()
    with urlopen(path, context=ctx) as resp:
        key_path.write_bytes(resp.read())
    print(f"Downloaded encryption key to {key_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BEHAVIOR-1K scene JSON + layout from Hugging Face (selective extract)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="OMNIGIBSON_DATA_PATH (default: <repo>/BEHAVIOR-1K/datasets).",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=Path("selected_scenes.txt"),
        help="Newline-separated scene names (default: ./selected_scenes.txt).",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Acknowledge dataset terms and allow downloading omnigibson.key.",
    )
    parser.add_argument(
        "--skip-key",
        action="store_true",
        help="Do not download omnigibson.key (scene graphs only need JSON + PNG).",
    )
    parser.add_argument(
        "--zip-version",
        default=BEHAVIOR_1K_DATASET_VERSION,
        help=f"Dataset zip version suffix (default: {BEHAVIOR_1K_DATASET_VERSION}).",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Use this file as the behavior-1k-assets zip (skip hf_hub_download).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Always call hf_hub_download (ignore recoverable local .incomplete blob).",
    )
    parser.add_argument(
        "--purge-stale-blob",
        action="store_true",
        help=(
            "Delete known corrupt HF cache blob (full-sized .incomplete that fails ZIP entry reads) "
            "and exit; then re-run without this flag to re-download."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = script_dir / "BEHAVIOR-1K" / "datasets"
    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    scene_list_path = args.scene_list
    if not scene_list_path.is_absolute():
        scene_list_path = (script_dir / scene_list_path).resolve()
    scenes = read_scene_list(scene_list_path if scene_list_path.is_file() else None)
    scene_set = set(scenes)

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.purge_stale_blob:
        stale = hf_behavior_blobs_dir() / f"{HF_BEHAVIOR_ZIP_ETAG}.incomplete"
        if stale.is_file():
            stale.unlink()
            log(f"Removed stale blob: {stale}")
        else:
            log(f"No file to remove: {stale}")
        locks = hf_behavior_blobs_dir().parent / ".locks"
        if locks.is_dir():
            for c in locks.iterdir():
                try:
                    c.unlink()
                except OSError:
                    pass
            log(f"Cleared locks under {locks}")
        log("Re-run: python3 download_scene_assets.py --accept-license ...")
        return

    log(f"Data directory: {data_dir}")
    log(f"Scenes to extract ({len(scenes)}): {', '.join(scenes)}")

    zip_filename = f"behavior-1k-assets-{args.zip_version}.zip"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        sys.exit(f"Install huggingface_hub: pip install huggingface-hub  ({e})")

    names: list[str] = []
    zip_path: Path | None = None
    if args.zip_path is not None:
        zip_path = args.zip_path.expanduser().resolve()
        if not zip_path.is_file():
            sys.exit(f"--zip-path does not exist: {zip_path}")
        if not zipfile.is_zipfile(zip_path):
            sys.exit(f"--zip-path is not a valid ZIP: {zip_path}")
        log(f"Using --zip-path: {zip_path}")
    elif not args.force_download and args.zip_version == BEHAVIOR_1K_DATASET_VERSION:
        zip_path = find_recoverable_local_zip(log)

    if zip_path is None:
        log(f"Downloading {zip_filename} from {HF_REPO} (large file; uses HF cache; resumes if partial)...")
        zip_path = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                filename=zip_filename,
                repo_type="dataset",
            )
        )
        log(f"Zip path: {zip_path}")
    else:
        log(f"Zip path: {zip_path}")

    assets_root = data_dir / "behavior-1k-assets"
    assets_root.mkdir(parents=True, exist_ok=True)

    key_path = data_dir / "omnigibson.key"
    fernet_obj = None
    if key_path.is_file():
        if Fernet is None:
            log("[warn] cryptography not installed; cannot decrypt entries. pip install cryptography")
        else:
            fernet_obj = Fernet(key_path.read_bytes())
            log(f"Using Fernet key from {key_path} for encrypted zip entries.")
    elif not args.skip_key and args.accept_license:
        log(f"\n{EULA_BLURB}")
        download_omnigibson_key(key_path)
        if Fernet and key_path.is_file():
            fernet_obj = Fernet(key_path.read_bytes())
            log("Key installed; encrypted scene files will be decrypted on extract.")
    else:
        log(
            "[info] No omnigibson.key yet. If extract fails with 'Bad magic', run with "
            "--accept-license (omit --skip-key) once to fetch the key, then re-run."
        )

    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        to_extract = [n for n in names if member_is_scene_json_or_layout(n, scene_set)]
        log(f"Extracting {len(to_extract)} files from zip (json + layout only)...")
        for i, member in enumerate(to_extract):
            if (i + 1) % 500 == 0 or i == 0:
                log(f"  ... {i + 1}/{len(to_extract)}")
            try:
                extract_member_to_disk(zf, member, data_dir, assets_root, fernet_obj)
                extracted += 1
            except (zipfile.error, OSError, ValueError) as err:
                log(f"  [warn] skip {member}: {err}")
                skipped += 1

    log(f"Done. Extracted {extracted} files ({skipped} skipped).")

    # Verify *_best.json per scene
    missing_best: list[str] = []
    for s in scenes:
        json_dir = assets_root / "scenes" / s / "json"
        best = json_dir / f"{s}_best.json"
        if not best.is_file():
            # May live under behavior-1k-assets/... if zip used nested prefix
            alt = data_dir / "behavior-1k-assets" / "scenes" / s / "json" / f"{s}_best.json"
            if not alt.is_file():
                missing_best.append(s)

    if missing_best:
        print(
            "[warn] Missing *_best.json for some scenes (check zip layout): "
            + ", ".join(missing_best)
        )
        # List sample paths for debugging
        sample = [n for n in names if "scenes/" in n and n.endswith(".json")][:15]
        if sample:
            print("Sample JSON paths in zip:")
            for p in sample:
                print(f"    {p}")

    if not key_path.is_file() and args.skip_key:
        log(
            "\n[note] No omnigibson.key (--skip-key). OmniGibson needs the key for USD assets; "
            "re-run once with --accept-license and without --skip-key if you use the simulator."
        )

    scenes_root = assets_root / "scenes"
    if scenes_root.is_dir():
        print("\nNext step:")
        print(
            "  python build_scene_graphs.py --scenes-root",
            scenes_root,
            "--output-dir output/scene_graphs --overlay-mpp 0.01",
        )
    else:
        print("\n[warn] Expected", scenes_root, "— check zip layout vs extraction paths.")


if __name__ == "__main__":
    main()
