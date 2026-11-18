"""Method-agnostic Dev-only interaction environment with hard OnlineBudgetGuard enforcement."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from harness.budget_tracker import OnlineBudgetGuard
from harness.env_adapter import create_env_adapter
from harness.policy_protocol import JOINT_CONTROL_MODES, normalize_control_mode

try:
    import gymnasium as gym  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    _GYM_BASE: Any = gym.Env
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _GYM_BASE = object

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT_DIR / "protocol" / "tasks_manifest.json"
DEFAULT_LOCK = ROOT_DIR / "protocol" / "splits.lock.json"


class DevInteractionEnv(_GYM_BASE):
    """Gym-compatible online interaction wrapper locked strictly to the `dev` split.

    Any RL or iterative optimization method may use this wrapper for online training/validation
    without ever touching `sealed_id` or `sealed_ood` and under an explicit episode cap.
    """

    def __init__(
        self,
        *,
        task_key: str | None = None,
        env_backend: str = "libero",
        perception_mode: str = "rgbd_projection",
        noise_sigma_mm: float = 10.0,
        control_mode: str = "osc_delta_eef",
        max_online_episodes: int = 2000,
        seed: int = 0,
        allow_dev_privileged_info: bool = False,
        manifest_path: Path = DEFAULT_MANIFEST,
        lock_path: Path = DEFAULT_LOCK,
    ) -> None:
        super().__init__()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))

        self.tasks_cfg: dict[str, dict[str, Any]] = manifest["tasks"]
        all_dev = lock_data["splits"]["dev"]
        if task_key is not None:
            if task_key not in self.tasks_cfg:
                raise ValueError(f"Unknown task_key: {task_key!r}")
            self.dev_episodes = [ep for ep in all_dev if ep["task_key"] == task_key]
        else:
            self.dev_episodes = list(all_dev)

        if not self.dev_episodes:
            raise RuntimeError("No dev episodes matched selection.")

        self.task_key = task_key
        self.control_mode = normalize_control_mode(control_mode)
        self.allow_dev_privileged_info = bool(allow_dev_privileged_info)
        self.budget_guard = OnlineBudgetGuard(max_online_episodes=max_online_episodes)
        self.adapter = create_env_adapter(
            env_backend,
            perception_mode=perception_mode,
            noise_sigma_mm=noise_sigma_mm,
            control_mode=self.control_mode,
        )
        self.rng = random.Random(seed)
        self._current_ep: dict[str, Any] = self.dev_episodes[0]
        self._current_cfg: dict[str, Any] = self.tasks_cfg[self._current_ep["task_key"]]
        self._steps = 0
        self._max_steps = int(self._current_cfg["extended_steps"])
        self._consecutive_succ = 0
        self._hold_target = int(self._current_cfg.get("hold_success_steps", 1))

        if gym is not None and np is not None:
            act_dim = 8 if self.control_mode in JOINT_CONTROL_MODES else 7
            bound = 1.0 if self.control_mode in ("osc_delta_eef", "delta_joint_pos") else 10.0
            self.action_space = gym.spaces.Box(low=-bound, high=bound, shape=(act_dim,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32),
                    "joint_pos": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
                    "joint_vel": gym.spaces.Box(low=-20.0, high=20.0, shape=(7,), dtype=np.float32),
                    "agentview": gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                    "wrist": gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                }
            )

    def _format_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        imgs = obs.get("images") or {}
        if np is not None:
            state = np.asarray(obs["state"], dtype=np.float32)
            joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)
            joint_vel = np.asarray(obs["joint_vel"], dtype=np.float32)
            agent = np.asarray(imgs.get("agentview", np.zeros((256, 256, 3), dtype=np.uint8)), dtype=np.uint8)
            wrist = np.asarray(imgs.get("wrist", np.zeros((256, 256, 3), dtype=np.uint8)), dtype=np.uint8)
        else:
            state = list(obs["state"])
            joint_pos = list(obs["joint_pos"])
            joint_vel = list(obs["joint_vel"])
            agent = imgs.get("agentview", [[[0, 0, 0]]])
            wrist = imgs.get("wrist", [[[0, 0, 0]]])
        out: dict[str, Any] = {
            "state": state,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "agentview": agent,
            "wrist": wrist,
            "instruction": obs.get("instruction", ""),
            "task_key": obs.get("task_key", ""),
        }
        if "depth" in obs:
            out["depth"] = obs["depth"]
        if "keypoints" in obs:
            out["keypoints"] = obs["keypoints"]
        return out

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        self.budget_guard.consume_online_episode(steps=self._steps, count=1)
        self._current_ep = self.rng.choice(self.dev_episodes)
        self._current_cfg = self.tasks_cfg[self._current_ep["task_key"]]
        self._steps = 0
        self._max_steps = int(self._current_cfg["extended_steps"])
        self._consecutive_succ = 0
        self._hold_target = int(self._current_cfg.get("hold_success_steps", 1))
        obs = self.adapter.reset_episode(self._current_ep, self._current_cfg)
        info_out: dict[str, Any] = {
            "episode_id": self._current_ep["episode_id"],
            "task_key": self._current_ep["task_key"],
            "remaining_budget": self.budget_guard.remaining_episodes,
        }
        if self.allow_dev_privileged_info:
            info_out["dev_privileged"] = self.adapter.get_telemetry()
        return self._format_obs(obs), info_out

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        act_list = [float(x) for x in list(action)]
        obs, _raw_r, _done, info = self.adapter.step(act_list)
        substeps = int(info.get("substeps", 1))
        self._steps += substeps
        if "consecutive_success" in info:
            self._consecutive_succ = int(info["consecutive_success"])
        elif info.get("is_success"):
            self._consecutive_succ += 1
        else:
            self._consecutive_succ = 0
        terminated = (self._consecutive_succ >= self._hold_target) or (info.get("sustained_at_substep") is not None)
        truncated = (not terminated) and (self._steps >= self._max_steps)
        reward = 1.0 if terminated else 0.0
        step_info: dict[str, Any] = {"is_success": terminated, "steps": self._steps, "substeps": substeps}
        if self.allow_dev_privileged_info:
            step_info["dev_privileged"] = self.adapter.get_telemetry()
        return (
            self._format_obs(obs),
            reward,
            terminated,
            truncated,
            step_info,
        )

    def close(self) -> None:
        self.adapter.close()
