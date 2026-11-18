"""Generate and verify SHA-256 locked benchmark splits for libero-rsi-bench."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = PROTOCOL_DIR / "tasks_manifest.json"
DEFAULT_LOCK_PATH = PROTOCOL_DIR / "splits.lock.json"


def canonical_json_bytes(obj: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for stable cryptographic hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_sealed_sha256(sealed_id: list[dict[str, Any]], sealed_ood: list[dict[str, Any]]) -> str:
    """Compute SHA-256 digest over both sealed test splits."""
    payload = {"sealed_id": sealed_id, "sealed_ood": sealed_ood}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_splits(manifest: dict[str, Any]) -> dict[str, Any]:
    """Construct burned, dev, sealed_id, and sealed_ood episode tables from manifest."""
    tasks: dict[str, dict[str, Any]] = manifest["tasks"]
    ood_axes: list[str] = manifest["ood_axes"]
    ood_levels: list[int] = manifest.get("ood_levels", [2, 3, 4])

    burned: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    sealed_id: list[dict[str, Any]] = []
    sealed_ood: list[dict[str, Any]] = []

    burn_idx = 0
    dev_idx = 0
    sid_idx = 0
    sood_idx = 0

    for task_idx, (task_key, task_cfg) in enumerate(tasks.items()):
        base_fields = {
            "task_key": task_key,
            "suite": task_cfg["suite"],
            "task_id": task_cfg["task_id"],
            "clean_base_name": task_cfg["clean_base_name"],
            "instruction": task_cfg["instruction"],
        }

        # Burned: 2 dynamic smoke-test seeds (1000+ range) per task, never used in Dev or Sealed.
        for offset in range(2):
            seed = 1000 + task_idx * 10 + offset
            burned.append(
                {
                    **base_fields,
                    "episode_id": f"burn_{burn_idx:03d}",
                    "split": "burned",
                    "init_state_id": None,
                    "seed": seed,
                    "ood_axis": None,
                    "ood_level": 0,
                    "variant_slot": None,
                }
            )
            burn_idx += 1

        # Dev: 25 official init states [0..24] per task -> 100 episodes total.
        for init_id in range(0, 25):
            dev.append(
                {
                    **base_fields,
                    "episode_id": f"dev_{dev_idx:03d}",
                    "split": "dev",
                    "init_state_id": init_id,
                    "seed": 2000 + task_idx * 100 + init_id,
                    "ood_axis": None,
                    "ood_level": 0,
                    "variant_slot": None,
                }
            )
            dev_idx += 1

        # Sealed ID: 25 official init states [25..49] per task -> 100 episodes total.
        for init_id in range(25, 50):
            sealed_id.append(
                {
                    **base_fields,
                    "episode_id": f"sid_{sid_idx:03d}",
                    "split": "sealed_id",
                    "init_state_id": init_id,
                    "seed": 4000 + task_idx * 100 + init_id,
                    "ood_axis": None,
                    "ood_level": 0,
                    "variant_slot": None,
                }
            )
            sid_idx += 1

        # Sealed OOD: 4 axes * 7 deterministic LIBERO-Plus variant slots per task -> 112 episodes.
        for axis_idx, axis in enumerate(ood_axes):
            for slot in range(7):
                level = ood_levels[slot % len(ood_levels)]
                sealed_ood.append(
                    {
                        **base_fields,
                        "episode_id": f"sood_{sood_idx:03d}",
                        "split": "sealed_ood",
                        "init_state_id": None,
                        "seed": 10000 + task_idx * 1000 + axis_idx * 100 + slot,
                        "ood_axis": axis,
                        "ood_level": level,
                        "variant_slot": slot,
                    }
                )
                sood_idx += 1

    sealed_sha256 = compute_sealed_sha256(sealed_id, sealed_ood)
    return {
        "benchmark_name": manifest["benchmark_name"],
        "version": manifest["version"],
        "sealed_sha256": sealed_sha256,
        "counts": {
            "burned": len(burned),
            "dev": len(dev),
            "sealed_id": len(sealed_id),
            "sealed_ood": len(sealed_ood),
        },
        "splits": {
            "burned": burned,
            "dev": dev,
            "sealed_id": sealed_id,
            "sealed_ood": sealed_ood,
        },
    }


def verify_lock(lock_data: dict[str, Any]) -> None:
    """Validate lockfile counts, disjointness, and sealed SHA-256 integrity."""
    splits = lock_data["splits"]
    expected_counts = {"burned": 8, "dev": 100, "sealed_id": 100, "sealed_ood": 112}
    for name, expected in expected_counts.items():
        actual = len(splits.get(name, []))
        if actual != expected:
            raise ValueError(f"Split '{name}' count mismatch: expected {expected}, got {actual}")

    # Verify init_state_id disjointness between dev and sealed_id per task.
    dev_keys = {(ep["task_key"], ep["init_state_id"]) for ep in splits["dev"]}
    sid_keys = {(ep["task_key"], ep["init_state_id"]) for ep in splits["sealed_id"]}
    overlap = dev_keys & sid_keys
    if overlap:
        raise ValueError(f"Dev and sealed_id init states overlap: {sorted(overlap)[:5]}")

    # Verify seed disjointness across all splits.
    all_seeds: list[int] = []
    for name in ("burned", "dev", "sealed_id", "sealed_ood"):
        all_seeds.extend(ep["seed"] for ep in splits[name])
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Duplicate seeds found across splits")

    expected_sha = compute_sealed_sha256(splits["sealed_id"], splits["sealed_ood"])
    if lock_data.get("sealed_sha256") != expected_sha:
        raise ValueError(
            f"Sealed SHA-256 mismatch: lock={lock_data.get('sealed_sha256')} computed={expected_sha}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or verify locked benchmark splits.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--verify", action="store_true", help="Verify existing splits.lock.json")
    args = parser.parse_args()

    if args.verify:
        with args.out.open("r", encoding="utf-8") as f:
            lock_data = json.load(f)
        verify_lock(lock_data)
        print(
            f"VERIFIED {args.out}: counts={lock_data['counts']}, "
            f"sealed_sha256={lock_data['sealed_sha256']}"
        )
        return

    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    lock_data = build_splits(manifest)
    verify_lock(lock_data)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(
        f"GENERATED {args.out}: counts={lock_data['counts']}, "
        f"sealed_sha256={lock_data['sealed_sha256']}"
    )


if __name__ == "__main__":
    main()
