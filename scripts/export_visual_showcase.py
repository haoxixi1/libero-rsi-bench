#!/usr/bin/env python3
"""Export rich simulation videos (.mp4), animated GIFs (.gif), and 6-frame trajectory filmstrips (.png)
showcasing the trained IL policy across the 4 foundational LIBERO tasks (ID & OOD).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines.il_policy import ILChunkingPolicy
from harness.env_adapter import create_env_adapter, rollout_episode
from harness.video_recorder import resolve_ffmpeg_bin, save_episode_keyframes, write_mp4, write_rgb_png

TASK_KEYS = ("spatial_t0", "object_t0", "goal_t0", "long_t0")
TASK_TITLES = {
    "spatial_t0": "Spatial (spatial_t0): Pick Black Bowl -> Place on Plate",
    "object_t0": "Object (object_t0): Pick Alphabet Soup -> Place in Basket",
    "goal_t0": "Goal (goal_t0): Articulate & Open Middle Cabinet Drawer",
    "long_t0": "Long-Horizon (long_t0): Put Both Alphabet Soup & Tomato Sauce in Basket",
}


def write_gif_from_frames(gif_path: Path, frames: list[np.ndarray], *, fps: int = 12, stride: int = 2) -> Path | None:
    """Encode a subsampled frame sequence into an animated GIF using ffmpeg or imageio."""
    if not frames:
        return None
    sampled = frames[:: max(1, stride)]
    if frames[-1] is not sampled[-1]:
        sampled.append(frames[-1])

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg_bin()
    if ffmpeg_bin:
        h, w, _ = sampled[0].shape
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(gif_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for f in sampled:
            proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait() == 0 and gif_path.is_file():
            return gif_path

    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(gif_path), sampled, fps=fps, loop=0)
        return gif_path
    except Exception:
        return None


def compose_6stage_filmstrip(
    out_png: Path,
    frames: list[np.ndarray],
    *,
    num_stages: int = 6,
    pad_px: int = 6,
) -> Path | None:
    """Create a horizontal 6-stage trajectory filmstrip (0%, 20%, 40%, 60%, 80%, 100%) from dual-view frames."""
    if not frames:
        return None
    n = len(frames)
    indices = [min(n - 1, int(round(i * (n - 1) / max(1, num_stages - 1)))) for i in range(num_stages)]
    selected = [np.asarray(frames[idx], dtype=np.uint8) for idx in indices]

    # Each dual-view frame has shape (256, 516, 3) -> stack vertically or in a 2x3 grid for high readability
    # 2 rows x 3 columns grid of dual-view pairs:
    h, w, _ = selected[0].shape
    cols = 3
    rows = (num_stages + cols - 1) // cols
    canvas_h = rows * h + (rows + 1) * pad_px
    canvas_w = cols * w + (cols + 1) * pad_px
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    for i, img in enumerate(selected):
        r = i // cols
        c = i % cols
        y0 = pad_px + r * (h + pad_px)
        x0 = pad_px + c * (w + pad_px)
        # Draw a thin status border around each stage
        canvas[y0 : y0 + h, x0 : x0 + w] = img

    return write_rgb_png(out_png, canvas)


def compose_filmstrip_from_saved_keyframes(
    keyframes_dir: Path,
    out_dir: Path,
    episodes_jsonl: Path,
) -> list[dict[str, Any]]:
    """Build per-task comparison sheets directly from existing evaluation keyframes without re-running simulator."""
    if not episodes_jsonl.is_file() or not keyframes_dir.is_dir():
        return []

    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]

    records = [json.loads(line) for line in episodes_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []

    for task_key in TASK_KEYS:
        task_eps = [r for r in records if r.get("task_key") == task_key]
        if not task_eps:
            continue
        # Prefer a successful episode first so the showcase highlights completed task behavior, plus one failure if any
        succ_eps = [r for r in task_eps if r.get("success_1_5x")]
        chosen = succ_eps[0] if succ_eps else task_eps[0]
        ep_id = chosen["episode_id"]
        cluster = chosen["failure_cluster"]

        k_files = sorted(keyframes_dir.glob(f"{ep_id}_*.png"))
        if len(k_files) == 3 and cv2 is not None:
            imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in k_files]
            h, w, _ = imgs[0].shape
            pad = 8
            strip = np.full((h + 2 * pad, 3 * w + 4 * pad, 3), 24, dtype=np.uint8)
            for idx, im in enumerate(imgs):
                x0 = pad + idx * (w + pad)
                strip[pad : pad + h, x0 : x0 + w] = im
            out_png = out_dir / f"{task_key}_{ep_id}_{cluster}_3stage.png"
            write_rgb_png(out_png, strip)
            exported.append(
                {
                    "task_key": task_key,
                    "episode_id": ep_id,
                    "outcome": cluster,
                    "steps_taken": chosen.get("steps_taken"),
                    "strip_png": str(out_png),
                    "keyframes": [str(p) for p in k_files],
                }
            )
    return exported


def run_live_video_showcase(
    *,
    checkpoint_path: Path,
    out_dir: Path,
    split: str = "dev",
    episodes_per_task: int = 1,
    env_backend: str = "libero",
) -> list[dict[str, Any]]:
    """Run live simulation rollouts for the 4 core tasks and export MP4 videos, GIFs, and 6-stage filmstrips."""
    manifest = json.loads((ROOT_DIR / "protocol" / "tasks_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT_DIR / "protocol" / "splits.lock.json").read_text(encoding="utf-8"))
    split_key = {"test_id": "sealed_id", "test_ood": "sealed_ood"}.get(split, split)
    all_specs = lock["splits"][split_key]

    # If a previous run's episodes.jsonl exists, pick episodes that succeeded so the video showcase shows successful manipulation
    preferred_ids: dict[str, list[str]] = {}
    for cand_run in (
        ROOT_DIR / "runs" / f"il_act_50_{split.replace('test_', '')}" / "episodes.jsonl",
        ROOT_DIR / "runs" / "il_act_50_dev" / "episodes.jsonl",
    ):
        if cand_run.is_file():
            for line in cand_run.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("success_1_5x") and r.get("split") == split_key:
                    preferred_ids.setdefault(r["task_key"], []).append(r["episode_id"])

    policy = ILChunkingPolicy(checkpoint_path=checkpoint_path)
    env = create_env_adapter(
        env_backend,
        perception_mode="rgbd_projection",
        control_mode=policy.control_mode,
    )

    videos_dir = out_dir / "videos"
    gifs_dir = out_dir / "gifs"
    strips_dir = out_dir / "filmstrips"
    kf_dir = out_dir / "keyframes"
    for d in (videos_dir, gifs_dir, strips_dir, kf_dir):
        d.mkdir(parents=True, exist_ok=True)

    showcase_items: list[dict[str, Any]] = []
    try:
        for task_key in TASK_KEYS:
            task_specs = [ep for ep in all_specs if ep["task_key"] == task_key]
            pref = preferred_ids.get(task_key, [])
            if pref:
                # Reorder so known successful episodes come first
                task_specs.sort(key=lambda x: (0 if x["episode_id"] in pref else 1, x["episode_id"]))

            for ep_spec in task_specs[:episodes_per_task]:
                ep_id = ep_spec["episode_id"]
                task_cfg = manifest["tasks"][task_key]
                print(f"[SHOWCASE] Rolling out task={task_key} episode={ep_id} ...")

                # Intercept frames via env recording
                env.set_frame_recording(True)
                obs = env.reset_episode(ep_spec, task_cfg)
                policy.reset(
                    {
                        "episode_id": ep_id,
                        "task_key": task_key,
                        "instruction": obs.get("instruction", task_cfg["instruction"]),
                        "standard_steps": int(task_cfg["standard_steps"]),
                        "extended_steps": int(task_cfg["extended_steps"]),
                        "control_mode": policy.control_mode,
                    }
                )
                ext_steps = int(task_cfg["extended_steps"])
                std_steps = int(task_cfg["standard_steps"])
                hold_target = int(task_cfg.get("hold_success_steps", 1))
                steps_taken = 0
                consec = 0
                sustained_step: int | None = None
                q: list[list[float]] = []

                while steps_taken < ext_steps:
                    if not q:
                        raw_act, _ = policy.act(obs)
                        if isinstance(raw_act[0], (list, tuple)):
                            q = [[float(x) for x in a] for a in raw_act]
                        else:
                            q = [[float(x) for x in raw_act]]
                    act_vec = q.pop(0)
                    obs, _, _, info = env.step(act_vec)
                    steps_taken += 1
                    if info.get("is_success"):
                        consec += 1
                        if consec >= hold_target and sustained_step is None:
                            sustained_step = steps_taken
                            break
                    else:
                        consec = 0

                frames = [np.asarray(f, dtype=np.uint8) for f in env.pop_recorded_frames()]
                env.set_frame_recording(False)

                succ_1_5x = sustained_step is not None and sustained_step <= ext_steps
                succ_1_0x = sustained_step is not None and sustained_step <= std_steps
                outcome = "success" if succ_1_5x else "fail"
                tag = f"{task_key}_{ep_id}_{outcome}"

                mp4_path = videos_dir / f"{tag}.mp4"
                gif_path = gifs_dir / f"{tag}.gif"
                strip_path = strips_dir / f"{tag}_6stage.png"

                write_mp4(mp4_path, frames, fps=20)
                write_gif_from_frames(gif_path, frames, fps=12, stride=2)
                compose_6stage_filmstrip(strip_path, frames, num_stages=6)
                kfs = save_episode_keyframes(kf_dir, f"{task_key}_{ep_id}", frames, suffix=outcome)

                showcase_items.append(
                    {
                        "task_key": task_key,
                        "task_title": TASK_TITLES.get(task_key, task_key),
                        "instruction": obs.get("instruction", task_cfg["instruction"]),
                        "episode_id": ep_id,
                        "split": split,
                        "ood_axis": ep_spec.get("ood_axis"),
                        "success_1_0x": succ_1_0x,
                        "success_1_5x": succ_1_5x,
                        "steps_taken": steps_taken,
                        "num_frames": len(frames),
                        "mp4": str(mp4_path.relative_to(out_dir)),
                        "gif": str(gif_path.relative_to(out_dir)),
                        "filmstrip_6stage": str(strip_path.relative_to(out_dir)),
                        "keyframes": [str(p.relative_to(out_dir)) for p in kfs],
                    }
                )
    finally:
        env.close()

    # Write SHOWCASE.md index and manifest.json
    (out_dir / "showcase_manifest.json").write_text(
        json.dumps(showcase_items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_lines = [
        "# LIBERO-RSI-Bench: Imitation Learning (ACT) Visual Rollout Showcase",
        "",
        "Each rollout frame displays synchronized dual-camera RGB views: **Left = `agentview` (3rd-person workspace camera)** | **Right = `robot0_eye_in_hand` (wrist camera)**.",
        "",
    ]
    for item in showcase_items:
        badge = "SUCCESS (1.0x)" if item["success_1_0x"] else ("SUCCESS (1.5x)" if item["success_1_5x"] else "FAILURE")
        md_lines.extend(
            [
                f"## {item['task_title']}",
                f"- **Episode**: `{item['episode_id']}` (`split={item['split']}`, `ood_axis={item['ood_axis'] or 'none'}`)",
                f"- **Instruction**: *\"{item['instruction']}\"*",
                f"- **Outcome**: **{badge}** in **{item['steps_taken']} steps**",
                f"- **Simulation Video (H.264 MP4)**: [`{item['mp4']}`]({item['mp4']})",
                f"- **Animated Rollout (GIF)**: [`{item['gif']}`]({item['gif']})",
                f"- **6-Stage Trajectory Filmstrip (0% -> 20% -> 40% -> 60% -> 80% -> 100%)**:",
                f"  ![{item['task_key']} 6-stage filmstrip]({item['filmstrip_6stage']})",
                "",
            ]
        )
    (out_dir / "SHOWCASE.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[OK] Exported {len(showcase_items)} visual showcase items to {out_dir}")
    return showcase_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Export simulation videos, GIFs, and filmstrips for 4 core LIBERO tasks.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT_DIR / "checkpoints" / "il_act_50" / "latest.pt")
    parser.add_argument("--split", type=str, default="test_id", choices=["dev", "test_id", "test_ood"])
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--env-backend", type=str, default="libero", choices=["libero", "mock"])
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "runs" / "visual_showcase")
    args = parser.parse_args()

    run_live_video_showcase(
        checkpoint_path=args.checkpoint,
        out_dir=args.out,
        split=args.split,
        episodes_per_task=args.episodes_per_task,
        env_backend=args.env_backend,
    )


if __name__ == "__main__":
    main()
