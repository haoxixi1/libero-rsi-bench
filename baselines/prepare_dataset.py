"""Prepare the locked 4-task offline dataset from HuggingFaceVLA/libero (or synthetic mock data).

Enforces the strict BENCHMARK_SPEC.md rules:
- Only the 4 benchmark tasks (`spatial_t0`, `object_t0`, `goal_t0`, `long_t0`)
- At most `max_demos_per_task` trajectories per task (default: 50 for full, 20 for low-data ablation)
- Automatically matches tasks via `meta/tasks.parquet` instruction text and only downloads the
  parquet shards that contain those episodes.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_MANIFEST = ROOT_DIR / "protocol" / "tasks_manifest.json"
TASK_ORDER = ("spatial_t0", "object_t0", "goal_t0", "long_t0")


def _normalize_text(s: str) -> str:
    return " ".join(str(s).lower().strip().replace("_", " ").replace("-", " ").split())


def _decode_image_cell(cell: Any) -> Any:
    """Decode a LeRobot parquet image cell (dict with 'bytes', PIL Image, or numpy array) to uint8 (H,W,3)."""
    import numpy as np
    from PIL import Image

    if isinstance(cell, np.ndarray):
        if cell.ndim == 3 and cell.shape[0] == 3 and cell.shape[2] != 3:
            cell = np.transpose(cell, (1, 2, 0))
        return np.ascontiguousarray(cell, dtype=np.uint8)
    if isinstance(cell, dict):
        raw_bytes = cell.get("bytes")
        if raw_bytes is not None:
            with Image.open(io.BytesIO(raw_bytes)) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
        if cell.get("path"):
            with Image.open(cell["path"]) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
    if isinstance(cell, (bytes, bytearray)):
        with Image.open(io.BytesIO(cell)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    if hasattr(cell, "convert"):
        return np.asarray(cell.convert("RGB"), dtype=np.uint8)
    return np.asarray(cell, dtype=np.uint8)


def generate_mock_dataset(out_dir: Path, max_demos_per_task: int = 8) -> Path:
    """Generate a lightweight synthetic 4-task dataset using MockLiberoEnvAdapter + ScriptedSmokePolicy."""
    import numpy as np
    import torch

    from harness.env_adapter import create_env_adapter
    from harness.evaluate import ScriptedSmokePolicy

    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    tasks_cfg = manifest["tasks"]
    out_dir.mkdir(parents=True, exist_ok=True)

    env = create_env_adapter("mock", perception_mode="noisy_keypoints", control_mode="osc_delta_eef")
    policy = ScriptedSmokePolicy()

    summary_stats: dict[str, Any] = {"backend": "mock", "max_demos_per_task": max_demos_per_task, "tasks": {}}

    for task_id_idx, task_key in enumerate(TASK_ORDER):
        t_cfg = tasks_cfg[task_key]
        episodes_out: list[dict[str, Any]] = []
        total_frames = 0

        for ep_i in range(max_demos_per_task):
            ep_spec = {
                "episode_id": f"mock_demo_{task_key}_{ep_i:03d}",
                "task_key": task_key,
                "suite": t_cfg["suite"],
                "task_id": t_cfg["task_id"],
                "clean_base_name": t_cfg["clean_base_name"],
                "split": "train_demo",
                "seed": 10000 + task_id_idx * 100 + ep_i,
            }
            obs = env.reset_episode(ep_spec, t_cfg)
            policy.reset({"task_key": task_key, "instruction": t_cfg["instruction"]})

            states: list[list[float]] = []
            actions: list[list[float]] = []
            agent_imgs: list[Any] = []
            wrist_imgs: list[Any] = []

            for step_t in range(int(t_cfg["standard_steps"])):
                act, _ = policy.act(obs)
                # Render synthetic patterned 256x256 RGB frames correlated with state for visual smoke testing
                st = obs["state"]
                canvas_a = np.zeros((256, 256, 3), dtype=np.uint8)
                canvas_w = np.zeros((256, 256, 3), dtype=np.uint8)
                cx = int(np.clip((st[0] + 0.3) * 400, 10, 245))
                cy = int(np.clip((st[1] + 0.3) * 400, 10, 245))
                cz = int(np.clip((st[2] - 0.75) * 400, 10, 245))
                canvas_a[cy - 6 : cy + 6, cx - 6 : cx + 6] = [255, 80, 60]
                canvas_w[cz - 6 : cz + 6, cx - 6 : cx + 6] = [60, 180, 255]

                states.append([float(x) for x in st])
                actions.append([float(x) for x in act])
                agent_imgs.append(canvas_a)
                wrist_imgs.append(canvas_w)

                obs, _r, _d, info = env.step(act)
                if info.get("is_success"):
                    break

            episodes_out.append(
                {
                    "episode_index": ep_i,
                    "task_key": task_key,
                    "task_id_idx": task_id_idx,
                    "instruction": t_cfg["instruction"],
                    "states": torch.tensor(states, dtype=torch.float32),
                    "actions": torch.tensor(actions, dtype=torch.float32),
                    "agentview": torch.from_numpy(np.stack(agent_imgs, axis=0)),
                    "wrist": torch.from_numpy(np.stack(wrist_imgs, axis=0)),
                }
            )
            total_frames += len(states)

        out_pt = out_dir / f"{task_key}.pt"
        torch.save(episodes_out, out_pt)
        summary_stats["tasks"][task_key] = {
            "num_episodes": len(episodes_out),
            "num_frames": total_frames,
            "file": str(out_pt.name),
        }
        print(f"  -> Saved mock task {task_key}: {len(episodes_out)} episodes ({total_frames} frames) -> {out_pt}")

    meta_path = out_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(summary_stats, indent=2), encoding="utf-8")
    return meta_path


def prepare_lerobot_hf_dataset(
    out_dir: Path,
    *,
    repo_id: str = "HuggingFaceVLA/libero",
    local_raw_dir: Path | None = None,
    max_demos_per_task: int = 50,
) -> Path:
    """Download/filter the 4 benchmark tasks from HuggingFaceVLA/libero and save fast PyTorch shards."""
    from collections import defaultdict
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download

    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    tasks_cfg = manifest["tasks"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_file(rel_path: str) -> Path:
        if local_raw_dir is not None and (local_raw_dir / rel_path).is_file():
            return local_raw_dir / rel_path
        import os
        import shutil
        import subprocess

        # Check existing HuggingFace hub snapshot cache first
        hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        snap_root = hf_home / "hub" / f"datasets--{repo_id.replace('/', '--')}" / "snapshots"
        if snap_root.is_dir():
            for snap in snap_root.iterdir():
                cached_p = snap / rel_path
                if cached_p.is_file() and cached_p.stat().st_size > 1000:
                    return cached_p

        fast_dir = out_dir.parent / "hf_raw_cache"
        target_file = fast_dir / rel_path
        if target_file.is_file() and target_file.stat().st_size > 1000:
            return target_file
        if shutil.which("aria2c") and rel_path.startswith("data/"):
            target_file.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel_path}"
            cmd = [
                "aria2c",
                "-c",
                "-x",
                "4",
                "-s",
                "4",
                "-k",
                "1M",
                "--max-tries=10",
                "--retry-wait=3",
                "--timeout=20",
                "--connect-timeout=10",
                "--user-agent=Mozilla/5.0",
                "--console-log-level=warn",
                "-d",
                str(target_file.parent),
                "-o",
                target_file.name,
                url,
            ]
            for _ in range(3):
                res = subprocess.run(cmd, check=False)
                if res.returncode == 0 and target_file.is_file() and target_file.stat().st_size > 1000:
                    return target_file
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                repo_type="dataset",
            )
        )

    print(f"[1/4] Fetching metadata from {repo_id} ...")
    tasks_pq_path = _fetch_file("meta/tasks.parquet")
    episodes_pq_path = _fetch_file("meta/episodes/chunk-000/file-000.parquet")

    tasks_table = pq.read_table(tasks_pq_path)
    tasks_df = tasks_table.to_pylist()
    print(f"  -> Loaded meta/tasks.parquet ({len(tasks_df)} total LIBERO tasks)")

    # Match our 4 target tasks by instruction string
    target_norm_to_key: dict[str, str] = {
        _normalize_text(tasks_cfg[k]["instruction"]): k for k in TASK_ORDER
    }

    task_index_to_key: dict[int, str] = {}
    for row_idx, row in enumerate(tasks_df):
        # LeRobot v3 tasks.parquet stores task description either in a column or index
        cand_texts = [str(v) for v in row.values() if isinstance(v, str)]
        cand_ints = [int(v) for k, v in row.items() if "index" in k.lower() and isinstance(v, int)]
        t_idx = cand_ints[0] if cand_ints else row_idx
        for txt in cand_texts:
            norm_txt = _normalize_text(txt)
            for target_norm, t_key in target_norm_to_key.items():
                if norm_txt == target_norm or target_norm in norm_txt:
                    task_index_to_key[t_idx] = t_key

    # Fallback if parquet stores task string in pandas index name (`__index_level_0__`)
    if len(set(task_index_to_key.values())) < 4:
        for row_idx, row in enumerate(tasks_df):
            for k, v in row.items():
                norm_txt = _normalize_text(str(v))
                for target_norm, t_key in target_norm_to_key.items():
                    if target_norm in norm_txt:
                        t_idx = int(row.get("task_index", row_idx))
                        task_index_to_key[t_idx] = t_key

    matched_keys = set(task_index_to_key.values())
    print(f"  -> Matched task_index mapping: {task_index_to_key}")
    if len(matched_keys) < 4:
        raise RuntimeError(
            f"Expected to match all 4 tasks in meta/tasks.parquet, only matched {task_index_to_key}. "
            f"Sample rows: {tasks_df[:5]}"
        )

    print("[2/4] Filtering episodes in meta/episodes/chunk-000/file-000.parquet ...")
    ep_table = pq.read_table(episodes_pq_path)
    ep_rows = ep_table.to_pylist()

    selected_episodes_by_task: dict[str, set[int]] = defaultdict(set)
    ep_to_task_key: dict[int, str] = {}
    required_files: set[str] = set()

    # Map instruction text to task_key directly as well in case episodes table has `tasks` list column
    for row in ep_rows:
        ep_idx = int(row["episode_index"])
        t_key: str | None = None
        if "task_index" in row and int(row["task_index"]) in task_index_to_key:
            t_key = task_index_to_key[int(row["task_index"])]
        elif "tasks" in row and isinstance(row["tasks"], list):
            for t_str in row["tasks"]:
                norm_t = _normalize_text(str(t_str))
                for target_norm, tk in target_norm_to_key.items():
                    if target_norm in norm_t:
                        t_key = tk
                        break

        if t_key is None:
            continue
        if len(selected_episodes_by_task[t_key]) >= max_demos_per_task:
            continue

        selected_episodes_by_task[t_key].add(ep_idx)
        ep_to_task_key[ep_idx] = t_key

    for tk in TASK_ORDER:
        print(f"  -> Task {tk}: selected {len(selected_episodes_by_task[tk])} episodes (cap={max_demos_per_task})")

    target_ep_set = set(ep_to_task_key.keys())

    locked_shards_path = ROOT_DIR / "protocol" / "hf_matched_shards.json"
    if locked_shards_path.is_file():
        shard_data = json.loads(locked_shards_path.read_text(encoding="utf-8"))
        shard_ranges = shard_data.get("shard_ranges", {})
        required_files = set()
        for f_str, (lo, hi) in shard_ranges.items():
            if any(int(lo) <= ep <= int(hi) for ep in target_ep_set):
                required_files.add(f"data/chunk-000/file-{int(f_str):03d}.parquet")
        print(f"[2.5/4] Loaded {len(required_files)} exact matched shards from {locked_shards_path.name}!")
    else:
        raise RuntimeError(f"Missing {locked_shards_path}")

    sorted_files = sorted(required_files)
    print(f"[3/4] Downloading & extracting frames from {len(sorted_files)} exact matched parquet shard(s)...")

    raw_frames_by_ep: dict[int, list[dict[str, Any]]] = defaultdict(list)

    if local_raw_dir is None and len(sorted_files) > 1:
        from concurrent.futures import ThreadPoolExecutor

        print(f"  -> Parallel downloading {len(sorted_files)} shards with 3 workers...")
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(_fetch_file, sorted_files))

    for f_idx, rel_pq in enumerate(sorted_files):
        local_pq = _fetch_file(rel_pq)
        table = pq.read_table(local_pq)
        col_names = set(table.column_names)
        ep_col = table["episode_index"].to_pylist()

        # Quick skip if no target episodes in this parquet shard
        if not any(int(e) in target_ep_set for e in set(ep_col)):
            if str(local_pq).startswith(str(out_dir.parent / "hf_raw_cache")):
                local_pq.unlink(missing_ok=True)
            continue

        print(f"  -> [{f_idx + 1}/{len(sorted_files)}] Reading {rel_pq} ...")
        rows = table.to_pylist()
        for r in rows:
            e_idx = int(r["episode_index"])
            if e_idx not in target_ep_set:
                continue
            img_main = _decode_image_cell(r["observation.images.image"])
            img_wrist = _decode_image_cell(
                r["observation.images.image2"] if "observation.images.image2" in col_names else r["observation.images.image"]
            )
            raw_frames_by_ep[e_idx].append(
                {
                    "frame_index": int(r.get("frame_index", len(raw_frames_by_ep[e_idx]))),
                    "state": [float(x) for x in r["observation.state"]],
                    "action": [float(x) for x in r["action"]],
                    "agentview": img_main,
                    "wrist": img_wrist,
                }
            )
        del rows, table
        if str(local_pq).startswith(str(out_dir.parent / "hf_raw_cache")):
            local_pq.unlink(missing_ok=True)

    print("[4/4] Saving per-task PyTorch dataset shards...")
    summary_stats: dict[str, Any] = {
        "backend": "lerobot_hf",
        "repo_id": repo_id,
        "max_demos_per_task": max_demos_per_task,
        "tasks": {},
    }

    for task_id_idx, task_key in enumerate(TASK_ORDER):
        t_cfg = tasks_cfg[task_key]
        ep_indices = sorted(selected_episodes_by_task[task_key])
        episodes_out: list[dict[str, Any]] = []
        total_frames = 0

        for local_i, e_idx in enumerate(ep_indices):
            frames = raw_frames_by_ep.get(e_idx, [])
            if not frames:
                continue
            frames.sort(key=lambda x: x["frame_index"])
            states = torch.tensor([f["state"] for f in frames], dtype=torch.float32)
            actions = torch.tensor([f["action"] for f in frames], dtype=torch.float32)
            agentview = torch.from_numpy(np.stack([f["agentview"] for f in frames], axis=0))
            wrist = torch.from_numpy(np.stack([f["wrist"] for f in frames], axis=0))

            episodes_out.append(
                {
                    "episode_index": int(e_idx),
                    "local_index": local_i,
                    "task_key": task_key,
                    "task_id_idx": task_id_idx,
                    "instruction": t_cfg["instruction"],
                    "states": states,
                    "actions": actions,
                    "agentview": agentview,
                    "wrist": wrist,
                }
            )
            total_frames += len(frames)

        out_pt = out_dir / f"{task_key}.pt"
        torch.save(episodes_out, out_pt)
        summary_stats["tasks"][task_key] = {
            "num_episodes": len(episodes_out),
            "num_frames": total_frames,
            "file": str(out_pt.name),
        }
        print(f"  -> Saved {task_key}: {len(episodes_out)} episodes, {total_frames} frames -> {out_pt}")

    meta_path = out_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(summary_stats, indent=2), encoding="utf-8")
    print(f"[OK] Dataset preparation complete! Metadata saved to {meta_path}")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 4-task offline dataset for IL experiments.")
    parser.add_argument("--source", choices=("hf", "mock"), default="hf", help="Data source ('hf' or 'mock')")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero", help="HuggingFace dataset repo ID")
    parser.add_argument("--local-raw-dir", type=Path, default=None, help="Optional local clone of HuggingFaceVLA/libero")
    parser.add_argument("--max-demos-per-task", type=int, default=50, help="Cap demos per task (50 full, 20 low-data)")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "datasets" / "il_4tasks_50", help="Output directory")
    args = parser.parse_args()

    if args.source == "mock":
        generate_mock_dataset(args.out_dir, max_demos_per_task=args.max_demos_per_task)
    else:
        prepare_lerobot_hf_dataset(
            args.out_dir,
            repo_id=args.repo_id,
            local_raw_dir=args.local_raw_dir,
            max_demos_per_task=args.max_demos_per_task,
        )


if __name__ == "__main__":
    main()
