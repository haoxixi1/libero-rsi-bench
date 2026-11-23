"""Fault-Tolerant Imitation Learning (ACT) Training Script for libero-rsi-bench.

Features:
- Loads pre-extracted 4-task dataset shards from `datasets/il_4tasks_50` (or `il_4tasks_20`).
- Supports both `osc_delta_eef` (7D relative delta) and `abs_eef` (7D target pose) training targets.
- Includes random image shift/crop & brightness/contrast augmentation for OOD camera/layout robustness.
- Mixed-precision (`bf16`/`fp16`) training optimized for RTX 5090.
- Automatic `latest.pt` / `best.pt` checkpoint saving, SIGINT/SIGTERM emergency checkpointing,
  and `--resume` support for `scripts/job_manager.sh`.
- Built-in periodic `dev` split evaluation option (`--eval-dev-every`) to select the best checkpoint
  by real `SR@1.5x` on the `dev` split!
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines.il_policy import TASK_KEY_TO_IDX, TASK_ORDER, build_il_network


def build_training_samples(
    data_dir: Path,
    *,
    control_mode: str = "osc_delta_eef",
    chunk_size: int = 16,
    max_demos_per_task: int | None = None,
    task_filter: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]], dict[str, list[float]]]:
    """Load per-task `.pt` episode shards and build (episode_idx, frame_t) index & normalization stats."""
    import torch

    episodes: list[dict[str, Any]] = []
    sample_indices: list[tuple[int, int]] = []
    all_states: list[Any] = []
    all_actions: list[Any] = []

    tasks_to_load = [t for t in TASK_ORDER if not task_filter or t in task_filter]
    for task_key in tasks_to_load:
        pt_path = data_dir / f"{task_key}.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(
                f"Missing dataset shard {pt_path}. Run `python3 baselines/prepare_dataset.py` first."
            )
        task_eps = torch.load(pt_path, map_location="cpu", weights_only=False)
        if max_demos_per_task is not None and max_demos_per_task > 0:
            task_eps = task_eps[:max_demos_per_task]

        for ep in task_eps:
            states = ep["states"].float()
            raw_actions = ep["actions"].float()
            n_frames = states.shape[0]
            if n_frames < 2:
                continue

            if control_mode == "abs_eef":
                # Next-step EEF pose (6D) + current gripper command (1D) per BENCHMARK_SPEC.md
                next_eef = torch.cat([states[1:, :6], states[-1:, :6]], dim=0)
                grip = raw_actions[:, 6:7]
                target_actions = torch.cat([next_eef, grip], dim=-1)
            else:
                target_actions = raw_actions

            ep_record = {
                "task_idx": TASK_KEY_TO_IDX.get(task_key, int(ep.get("task_id_idx", 0))),
                "states": states,
                "actions": target_actions,
                "agentview": ep["agentview"],
                "wrist": ep["wrist"],
            }
            ep_global_idx = len(episodes)
            episodes.append(ep_record)
            all_states.append(states)
            all_actions.append(target_actions)
            for t in range(n_frames):
                sample_indices.append((ep_global_idx, t))

    cat_states = torch.cat(all_states, dim=0)
    cat_actions = torch.cat(all_actions, dim=0)

    state_mean = cat_states.mean(dim=0)
    state_std = cat_states.std(dim=0).clamp(min=1e-4)
    action_mean = cat_actions.mean(dim=0)
    action_std = cat_actions.std(dim=0).clamp(min=1e-4)
    # Keep gripper dimension unscaled (mean=0, std=1) so ±1 sign stays crisp
    action_mean[6] = 0.0
    action_std[6] = 1.0

    norm_stats = {
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
    }
    return episodes, sample_indices, norm_stats


def augment_image_batch(img_bchw: Any, training: bool = True) -> Any:
    """Apply fast GPU random translation (pad-and-crop) and mild brightness/contrast jitter."""
    import torch
    import torch.nn.functional as F

    if not training:
        return img_bchw
    b, c, h, w = img_bchw.shape
    padded = F.pad(img_bchw, (6, 6, 6, 6), mode="replicate")
    dy = int(torch.randint(0, 13, (1,)).item())
    dx = int(torch.randint(0, 13, (1,)).item())
    cropped = padded[:, :, dy : dy + h, dx : dx + w]

    # Mild brightness & contrast perturbation for camera_visual OOD resilience
    scale = 1.0 + (torch.rand((b, 1, 1, 1), device=img_bchw.device) - 0.5) * 0.20
    bias = (torch.rand((b, 1, 1, 1), device=img_bchw.device) - 0.5) * 0.12
    return (cropped * scale + bias).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dual-Camera ACT Imitation Learning Policy.")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "datasets" / "il_4tasks_50")
    parser.add_argument("--save-dir", type=Path, default=ROOT_DIR / "checkpoints" / "il_act_50")
    parser.add_argument("--policy-id", type=str, default="il_act_50")
    parser.add_argument("--control-mode", choices=("osc_delta_eef", "abs_eef"), default="osc_delta_eef")
    parser.add_argument("--max-demos-per-task", type=int, default=50, help="50 for full, 20 for low-data ablation")
    parser.add_argument("--task-filter", nargs="*", default=None, help="Optional subset of tasks")
    parser.add_argument("--backbone", choices=("resnet18", "cnn"), default="resnet18")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100, help="BENCHMARK_SPEC cap: max 100 epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Auto-resume from save_dir/latest.pt if present")
    parser.add_argument("--eval-dev-every", type=int, default=0, help="Optional dev evaluation interval (epochs)")
    parser.add_argument("--eval-backend", choices=("mock", "libero"), default="libero")
    parser.add_argument("--eval-limit", type=int, default=20, help="Episodes on dev split during periodic check")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading dataset from {args.data_dir} (max_demos_per_task={args.max_demos_per_task})...")
    episodes, sample_indices, norm_stats = build_training_samples(
        args.data_dir,
        control_mode=args.control_mode,
        chunk_size=args.chunk_size,
        max_demos_per_task=args.max_demos_per_task,
        task_filter=args.task_filter,
    )
    total_demos = len(episodes)
    total_samples = len(sample_indices)
    print(f"  -> Loaded {total_demos} episodes ({total_samples} frame transitions) on device={device}")

    # Hold out 5% of trajectories (at least 1 per task if >=4 episodes) for validation loss tracking
    rng = random.Random(args.seed)
    val_ep_set = set(rng.sample(range(total_demos), max(1, total_demos // 15))) if total_demos >= 8 else set()
    train_indices = [pair for pair in sample_indices if pair[0] not in val_ep_set]
    val_indices = [pair for pair in sample_indices if pair[0] in val_ep_set] or train_indices[: min(256, len(train_indices))]

    state_mean = torch.tensor(norm_stats["state_mean"], dtype=torch.float32)
    state_std = torch.tensor(norm_stats["state_std"], dtype=torch.float32)
    action_mean = torch.tensor(norm_stats["action_mean"], dtype=torch.float32)
    action_std = torch.tensor(norm_stats["action_std"], dtype=torch.float32)

    class ChunkDataset(Dataset):
        def __init__(self, indices: list[tuple[int, int]]) -> None:
            self.indices = indices

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx: int) -> tuple[Any, Any, Any, Any, Any]:
            ep_i, t = self.indices[idx]
            ep = episodes[ep_i]
            T = ep["states"].shape[0]

            agent_img = F.interpolate(
                ep["agentview"][t].permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            wrist_img = F.interpolate(
                ep["wrist"][t].permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            st_norm = (ep["states"][t] - state_mean) / state_std

            # Extract action chunk [t .. t + chunk_size - 1], padding after episode end with final hold action
            end_t = min(T, t + args.chunk_size)
            chunk = ep["actions"][t:end_t]
            if chunk.shape[0] < args.chunk_size:
                pad_len = args.chunk_size - chunk.shape[0]
                last_a = ep["actions"][-1:].clone()
                if args.control_mode == "osc_delta_eef":
                    last_a[:, :6] = 0.0
                chunk = torch.cat([chunk, last_a.expand(pad_len, -1)], dim=0)
            chunk_norm = (chunk - action_mean) / action_std
            return agent_img, wrist_img, st_norm, torch.tensor(ep["task_idx"], dtype=torch.long), chunk_norm

    num_workers = 4 if device == "cuda" else 0
    train_loader = DataLoader(
        ChunkDataset(train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        ChunkDataset(val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"[2/3] Building Dual-Camera ACT model (backbone={args.backbone}, chunk_size={args.chunk_size})...")
    model = build_il_network(
        state_dim=8,
        action_dim=7,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        backbone=args.backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps_cap = min(args.epochs * max(1, len(train_loader)), 50000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    best_dev_sr = -1.0
    cum_gpu_seconds = 0.0

    latest_ckpt = args.save_dir / "latest.pt"
    best_ckpt = args.save_dir / "best.pt"
    log_jsonl = args.save_dir / "train_log.jsonl"

    if args.resume and latest_ckpt.is_file():
        print(f"  -> Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_dev_sr = float(ckpt.get("best_dev_sr", -1.0))
        cum_gpu_seconds = float(ckpt.get("cum_gpu_seconds", 0.0))
        print(f"  -> Resumed at epoch {start_epoch}/{args.epochs} (global_step={global_step}, gpu_hours={cum_gpu_seconds/3600:.2f})")

    model_config = {
        "policy_id": args.policy_id,
        "control_mode": args.control_mode,
        "state_dim": 8,
        "action_dim": 7,
        "chunk_size": args.chunk_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "backbone": args.backbone,
        "demo_count": total_demos,
    }

    def save_checkpoint(path: Path, epoch_num: int, val_loss_val: float, dev_sr_val: float) -> None:
        tmp_path = path.with_suffix(".tmp.pt")
        payload = {
            "epoch": epoch_num,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_dev_sr": best_dev_sr,
            "val_loss": val_loss_val,
            "dev_sr": dev_sr_val,
            "cum_gpu_seconds": cum_gpu_seconds,
            "gpu_hours": round(cum_gpu_seconds / 3600.0, 4),
            "demo_count": total_demos,
            "config": model_config,
            "norm_stats": norm_stats,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    # Register graceful SIGINT/SIGTERM emergency save
    interrupted = {"flag": False}

    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"\n[SIGNAL {signum}] Caught interrupt! Saving emergency checkpoint to {latest_ckpt} ...")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"[3/3] Starting training for epochs {start_epoch}..{args.epochs} (50k step protocol cap)...")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for agent_b, wrist_b, state_b, task_b, chunk_b in train_loader:
            if interrupted["flag"] or global_step >= 50000:
                break
            agent_b = augment_image_batch(agent_b.to(device, non_blocking=True), training=True)
            wrist_b = augment_image_batch(wrist_b.to(device, non_blocking=True), training=True)
            state_b = state_b.to(device, non_blocking=True)
            task_b = task_b.to(device, non_blocking=True)
            chunk_b = chunk_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(agent_b, wrist_b, state_b, task_b)
                loss_pos = F.smooth_l1_loss(pred[:, :, :6], chunk_b[:, :, :6], beta=0.05)
                loss_grip = F.mse_loss(pred[:, :, 6:], chunk_b[:, :, 6:])
                loss = loss_pos + 0.5 * loss_grip

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1
            global_step += 1

        scheduler.step()

        # Validation pass
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for agent_b, wrist_b, state_b, task_b, chunk_b in val_loader:
                agent_b = agent_b.to(device, non_blocking=True)
                wrist_b = wrist_b.to(device, non_blocking=True)
                state_b = state_b.to(device, non_blocking=True)
                task_b = task_b.to(device, non_blocking=True)
                chunk_b = chunk_b.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    pred = model(agent_b, wrist_b, state_b, task_b)
                    v_loss = F.smooth_l1_loss(pred[:, :, :6], chunk_b[:, :, :6], beta=0.05) + 0.5 * F.mse_loss(
                        pred[:, :, 6:], chunk_b[:, :, 6:]
                    )
                val_loss_sum += float(v_loss.item())
                val_batches += 1

        train_loss = train_loss_sum / max(1, train_batches)
        val_loss = val_loss_sum / max(1, val_batches)
        elapsed = time.time() - t0
        cum_gpu_seconds += elapsed

        dev_sr = -1.0
        save_checkpoint(latest_ckpt, epoch, val_loss, dev_sr)
        if not best_ckpt.is_file():
            save_checkpoint(best_ckpt, epoch, val_loss, dev_sr)

        if args.eval_dev_every > 0 and (epoch % args.eval_dev_every == 0 or epoch == args.epochs):
            from harness.evaluate import run_evaluation

            dev_out = args.save_dir / f"dev_eval_ep{epoch:03d}"
            summary = run_evaluation(
                split="dev",
                out_dir=dev_out,
                env_backend=args.eval_backend,
                policy_class="baselines.il_policy:ILChunkingPolicy",
                policy_kwargs=json.dumps({"checkpoint_path": str(latest_ckpt)}),
                limit=args.eval_limit,
                demo_count=total_demos,
                gpu_hours=round(cum_gpu_seconds / 3600.0, 3),
            )
            dev_sr = float(summary.get("sr_1_5x", 0.0))

        is_new_best = False
        if dev_sr >= 0.0:
            if dev_sr > best_dev_sr or (dev_sr == best_dev_sr and val_loss < best_val_loss):
                best_dev_sr = dev_sr
                best_val_loss = min(best_val_loss, val_loss)
                is_new_best = True
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            is_new_best = True

        if is_new_best:
            save_checkpoint(best_ckpt, epoch, val_loss, dev_sr)

        rec = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "dev_sr_1_5x": round(dev_sr, 4) if dev_sr >= 0 else None,
            "best_val_loss": round(best_val_loss, 6),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "epoch_sec": round(elapsed, 2),
            "gpu_hours": round(cum_gpu_seconds / 3600.0, 4),
            "demo_count": total_demos,
            "is_best": is_new_best,
        }
        with log_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        star = " *BEST*" if is_new_best else ""
        dev_str = f" | dev_SR@1.5x={dev_sr*100:.1f}%" if dev_sr >= 0 else ""
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] step={global_step:05d} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}{dev_str} | "
            f"time={elapsed:.1f}s (total={cum_gpu_seconds/3600:.2f} GPUh){star}"
        )

        if interrupted["flag"] or global_step >= 50000:
            print(f"[STOP] Stopping at epoch {epoch} (interrupted={interrupted['flag']}, step={global_step}).")
            break

    # Save ledger metadata JSON so evaluation script automatically reads exact demo_count & gpu_hours
    ledger_meta = {
        "policy_id": args.policy_id,
        "demo_count": total_demos,
        "gpu_hours": round(max(0.01, cum_gpu_seconds / 3600.0), 3),
        "best_val_loss": round(best_val_loss, 6),
        "best_dev_sr": round(best_dev_sr, 4),
        "best_checkpoint": str(best_ckpt),
    }
    (args.save_dir / "ledger_meta.json").write_text(json.dumps(ledger_meta, indent=2), encoding="utf-8")
    print(f"[DONE] Best checkpoint saved to: {best_ckpt}")
    print(f"       Ledger metadata: {ledger_meta}")


if __name__ == "__main__":
    main()
