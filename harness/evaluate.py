"""Authoritative unified benchmark evaluator for libero-rsi-bench."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import time
from pathlib import Path
from typing import Any

from harness.budget_tracker import CostLedger, OnlineBudgetGuard
from harness.env_adapter import EpisodeResult, create_env_adapter, rollout_episode
from harness.math_kinematics import clip, vec2_dist
from harness.policy_protocol import (
    BasePolicy,
    CONTROL_MODE_ALIASES,
    RemotePolicyClient,
    VALID_CONTROL_MODES,
    normalize_control_mode,
)
from protocol.generate_splits import verify_lock

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT_DIR / "protocol" / "tasks_manifest.json"
DEFAULT_LOCK = ROOT_DIR / "protocol" / "splits.lock.json"


class ZeroDummyPolicy(BasePolicy):
    """Minimal no-op dummy policy for harness and simulator connectivity checks."""

    policy_id = "dummy_zero"
    family = "il"

    def reset(self, task_meta: dict[str, Any]) -> None:
        return

    def act(self, obs_dict: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], {}


class RandomDummyPolicy(BasePolicy):
    """Seeded random-action dummy policy for harness verification."""

    policy_id = "dummy_random"
    family = "rl"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def reset(self, task_meta: dict[str, Any]) -> None:
        return

    def act(self, obs_dict: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
        act = [self._rng.uniform(-0.3, 0.3) for _ in range(6)] + [self._rng.choice([-1.0, 1.0])]
        return act, {}


class ScriptedSmokePolicy(BasePolicy):
    """Lightweight mock-simulator smoke policy (approaches keypoints & completes pick-place/drawer)."""

    policy_id = "dummy_scripted_smoke"
    family = "agentic"

    def __init__(self) -> None:
        self._phase = "hover"
        self._hold = 0
        self._cycles = 0
        self._task_type = "pick_place"

    def reset(self, task_meta: dict[str, Any]) -> None:
        self._phase = "hover"
        self._hold = 0
        self._cycles = 0
        instr = str(task_meta.get("instruction", "")).lower()
        if "drawer" in instr or "cabinet" in instr:
            self._task_type = "articulated_drawer"
        elif "both" in instr or "along with" in instr or "tomato sauce" in instr:
            self._task_type = "multi_stage_pick_place"
        else:
            self._task_type = "pick_place"

    def act(self, obs_dict: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
        kp = obs_dict.get("keypoints") or {}
        eef = list(obs_dict.get("eef_pos", [0.0, 0.0, 1.0]))
        tgt = list(kp.get("target_xyz", [0.0, 0.0, 0.84]))
        goal = list(kp.get("goal_xyz", [0.15, 0.15, 0.84]))
        macro_inc = 0

        if self._task_type == "articulated_drawer":
            handle = list(kp.get("drawer_handle_xyz") or tgt)
            if self._phase == "hover":
                d = [clip(8.0 * (handle[i] - eef[i]), -1.0, 1.0) for i in range(3)]
                if max(abs(handle[i] - eef[i]) for i in range(3)) < 0.02:
                    self._phase = "grasp"
                    self._hold = 0
                    macro_inc = 1
                return d + [0.0, 0.0, 0.0, -1.0], {"macro_turn_increment": macro_inc}
            if self._phase == "grasp":
                self._hold += 1
                if self._hold >= 8:
                    self._phase = "pull"
                    macro_inc = 1
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], {"macro_turn_increment": macro_inc}
            pull_goal = [handle[0] - 0.20, handle[1], handle[2]]
            d = [clip(8.0 * (pull_goal[i] - eef[i]), -1.0, 1.0) for i in range(3)]
            return d + [0.0, 0.0, 0.0, 1.0], {"macro_turn_increment": 0}

        if self._phase == "hover":
            wp = [tgt[0], tgt[1], tgt[2] + 0.14]
            d = [clip(8.0 * (wp[i] - eef[i]), -1.0, 1.0) for i in range(3)]
            if max(abs(wp[i] - eef[i]) for i in range(3)) < 0.02:
                self._phase = "descend"
                macro_inc = 1
            return d + [0.0, 0.0, 0.0, -1.0], {"macro_turn_increment": macro_inc}
        if self._phase == "descend":
            d = [clip(8.0 * (tgt[i] - eef[i]), -1.0, 1.0) for i in range(3)]
            if max(abs(tgt[i] - eef[i]) for i in range(3)) < 0.02:
                self._phase = "grasp"
                self._hold = 0
                macro_inc = 1
            return d + [0.0, 0.0, 0.0, -1.0], {"macro_turn_increment": macro_inc}
        if self._phase == "grasp":
            self._hold += 1
            if self._hold >= 8:
                self._phase = "lift"
                macro_inc = 1
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], {"macro_turn_increment": macro_inc}
        if self._phase == "lift":
            if eef[2] >= goal[2] + 0.12:
                self._phase = "move"
                macro_inc = 1
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], {"macro_turn_increment": macro_inc}
        if self._phase == "move":
            wp = [goal[0], goal[1], goal[2] + 0.11]
            d = [clip(8.0 * (wp[i] - eef[i]), -1.0, 1.0) for i in range(3)]
            if vec2_dist(eef, goal) < 0.025:
                self._phase = "release"
                self._hold = 0
                macro_inc = 1
            return d + [0.0, 0.0, 0.0, 1.0], {"macro_turn_increment": macro_inc}
        self._hold += 1
        if self._hold >= 8 and self._task_type == "multi_stage_pick_place" and self._cycles == 0:
            self._cycles += 1
            self._phase = "hover"
            macro_inc = 1
        return [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0], {"macro_turn_increment": macro_inc}


def build_dummy_policy(name: str) -> BasePolicy:
    """Create a minimal built-in dummy policy strictly for harness testing."""
    if name == "zero":
        return ZeroDummyPolicy()
    if name == "random":
        return RandomDummyPolicy()
    if name == "scripted_smoke":
        return ScriptedSmokePolicy()
    raise ValueError(f"Unknown dummy policy: {name!r}")


def load_policy_from_class(spec: str, kwargs_json: str | None = None) -> BasePolicy:
    """Dynamically load a teammate's BasePolicy implementation via `module.path:ClassName`."""
    if ":" not in spec:
        raise ValueError(f"--policy-class must use 'module.path:ClassName', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    return cls(**kwargs)


def run_evaluation(
    *,
    split: str,
    out_dir: Path,
    env_backend: str = "mock",
    perception_mode: str = "rgbd_projection",
    noise_sigma_mm: float = 10.0,
    control_mode: str | None = None,
    policy_url: str | None = None,
    policy_class: str | None = None,
    policy_kwargs: str | None = None,
    dummy_policy: str | None = None,
    confirm_unseal: bool = False,
    demo_count: int = 0,
    gpu_hours: float = 0.0,
    manifest_path: Path = DEFAULT_MANIFEST,
    lock_path: Path = DEFAULT_LOCK,
    sealed_log_path: Path | None = None,
    task_filter: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate any remote HTTP or in-process BasePolicy on a locked split."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_lock(lock_data)

    if split not in lock_data["splits"]:
        raise ValueError(f"Invalid split {split!r}; expected one of {list(lock_data['splits'])}")

    episodes_spec = list(lock_data["splits"][split])
    if task_filter:
        allowed = set(task_filter)
        episodes_spec = [ep for ep in episodes_spec if ep["task_key"] in allowed]
    if limit is not None and limit > 0:
        episodes_spec = episodes_spec[:limit]

    if policy_url:
        policy: BasePolicy = RemotePolicyClient(policy_url)
    elif policy_class:
        policy = load_policy_from_class(policy_class, policy_kwargs)
    elif dummy_policy:
        policy = build_dummy_policy(dummy_policy)
    else:
        raise ValueError("Provide one of --policy-url, --policy-class, or --dummy-policy")

    guard = OnlineBudgetGuard(
        max_online_episodes=int(manifest["budgets"]["online_dev_episodes_total"]),
        sealed_log_path=sealed_log_path,
    )
    guard.check_split_access(
        split,
        allow_unseal=confirm_unseal,
        policy_id=policy.policy_id,
        sealed_sha256=lock_data["sealed_sha256"],
        num_episodes=len(episodes_spec),
    )

    ledger = CostLedger(demo_count=demo_count, gpu_hours=gpu_hours)
    effective_control_mode = normalize_control_mode(control_mode or getattr(policy, "control_mode", "osc_delta_eef"))
    env = create_env_adapter(
        env_backend,
        perception_mode=perception_mode,
        noise_sigma_mm=noise_sigma_mm,
        control_mode=effective_control_mode,
    )

    t0 = time.monotonic()
    results: list[EpisodeResult] = []
    try:
        tasks_cfg = manifest["tasks"]
        for ep in episodes_spec:
            res = rollout_episode(env, policy, ep, tasks_cfg[ep["task_key"]], ledger=ledger)
            results.append(res)
    finally:
        env.close()

    ledger.eval_wall_sec = round(time.monotonic() - t0, 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    ep_lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in results]
    (out_dir / "episodes.jsonl").write_text("\n".join(ep_lines) + "\n", encoding="utf-8")

    n = len(results)
    k1 = sum(1 for r in results if r.success_1_0x)
    k2 = sum(1 for r in results if r.success_1_5x)

    per_task: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = per_task.setdefault(r.task_key, {"n": 0, "sr_1_0x": 0, "sr_1_5x": 0})
        bucket["n"] += 1
        bucket["sr_1_0x"] += int(r.success_1_0x)
        bucket["sr_1_5x"] += int(r.success_1_5x)
    for bucket in per_task.values():
        cnt = max(1, bucket["n"])
        bucket["sr_1_0x_rate"] = bucket["sr_1_0x"] / cnt
        bucket["sr_1_5x_rate"] = bucket["sr_1_5x"] / cnt

    per_ood_axis: dict[str, dict[str, Any]] = {}
    for r in results:
        if r.ood_axis:
            ax = per_ood_axis.setdefault(r.ood_axis, {"n": 0, "sr_1_0x": 0, "sr_1_5x": 0})
            ax["n"] += 1
            ax["sr_1_0x"] += int(r.success_1_0x)
            ax["sr_1_5x"] += int(r.success_1_5x)
    for ax in per_ood_axis.values():
        cnt = max(1, ax["n"])
        ax["sr_1_0x_rate"] = ax["sr_1_0x"] / cnt
        ax["sr_1_5x_rate"] = ax["sr_1_5x"] / cnt

    summary = {
        "policy_id": policy.policy_id,
        "family": policy.family,
        "control_mode": effective_control_mode,
        "split": split,
        "env_backend": env_backend,
        "perception_mode": perception_mode,
        "noise_sigma_mm": noise_sigma_mm,
        "sealed_sha256": lock_data["sealed_sha256"],
        "num_episodes": n,
        "success_1_0x_count": k1,
        "success_1_5x_count": k2,
        "sr_1_0x": k1 / max(1, n),
        "sr_1_5x": k2 / max(1, n),
        "mean_steps": sum(r.steps_taken for r in results) / max(1, n),
        "mean_macro_turns": sum(r.macro_turns for r in results) / max(1, n),
        "per_task": per_task,
        "per_ood_axis": per_ood_axis,
        "cost_ledger": ledger.to_dict(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified libero-rsi-bench evaluation.")
    parser.add_argument("--split", choices=("burned", "dev", "sealed_id", "sealed_ood"), required=True)
    parser.add_argument("--env-backend", choices=("mock", "libero"), default="mock")
    parser.add_argument("--perception-mode", choices=("rgbd_projection", "noisy_keypoints"), default="rgbd_projection")
    parser.add_argument("--noise-sigma-mm", type=float, default=10.0)
    parser.add_argument(
        "--control-mode",
        choices=VALID_CONTROL_MODES + tuple(CONTROL_MODE_ALIASES),
        default=None,
        help="Override policy-declared control_mode (defaults to /health control_mode)",
    )
    parser.add_argument("--policy-url", type=str, default=None, help="HTTP Policy Server URL (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--policy-class", type=str, default=None, help="Import path 'module:ClassName' implementing BasePolicy")
    parser.add_argument("--policy-kwargs", type=str, default=None, help="JSON kwargs for --policy-class")
    parser.add_argument("--dummy-policy", choices=("zero", "random", "scripted_smoke"), default=None)
    parser.add_argument("--confirm-unseal", action="store_true", help="Authorize one-time sealed split evaluation")
    parser.add_argument("--demo-count", type=int, default=0)
    parser.add_argument("--gpu-hours", type=float, default=0.0)
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional subset of task_keys for smoke checks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = run_evaluation(
        split=args.split,
        out_dir=args.out,
        env_backend=args.env_backend,
        perception_mode=args.perception_mode,
        noise_sigma_mm=args.noise_sigma_mm,
        control_mode=args.control_mode,
        policy_url=args.policy_url,
        policy_class=args.policy_class,
        policy_kwargs=args.policy_kwargs,
        dummy_policy=args.dummy_policy,
        confirm_unseal=args.confirm_unseal,
        demo_count=args.demo_count,
        gpu_hours=args.gpu_hours,
        task_filter=args.tasks,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
