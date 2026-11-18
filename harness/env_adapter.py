"""Unified LIBERO / LIBERO-Plus environment adapter, mock simulator, and rollout runner."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from harness.budget_tracker import CostLedger
from harness.math_kinematics import (
    axisangle2quat,
    clip,
    deterministic_gaussian_vec3,
    quat2axisangle,
    quat_multiply,
    quat_orientation_error_aa,
    vec2_dist,
    vec3_add,
    vec3_dist,
    vec3_norm,
    vec3_sub,
)
from harness.policy_protocol import (
    BasePolicy,
    JOINT_CONTROL_MODES,
    MACRO_CONTROL_MODES,
    normalize_control_mode,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


FAILURE_CLUSTERS = (
    "success",
    "approach_miss",
    "grasp_slip",
    "place_or_articulate_fail",
    "timeout_incomplete",
)

PARAPHRASED_INSTRUCTIONS = {
    "spatial_t0": [
        "grab the black bowl sitting between the plate and ramekin and put it onto the plate",
        "move the dark bowl positioned near the ramekin onto the serving plate",
        "place the black bowl from between the two dishes onto the plate",
    ],
    "object_t0": [
        "lift the alphabet soup can and drop it inside the basket",
        "transfer the alphabet soup tin into the storage basket",
        "pick the soup container up and set it into the basket",
    ],
    "goal_t0": [
        "slide open the center drawer of the wooden cabinet",
        "pull the middle cabinet drawer outward",
        "open up the cabinet's middle drawer",
    ],
    "long_t0": [
        "place the alphabet soup and the tomato sauce cans into the basket",
        "put the soup can along with the tomato sauce container inside the basket",
        "move both cans—alphabet soup and tomato sauce—into the basket",
    ],
}


@dataclass
class EpisodeResult:
    """Standardized per-episode evaluation record across all 4 method families."""

    episode_id: str
    task_key: str
    split: str
    ood_axis: str | None
    success_1_0x: bool
    success_1_5x: bool
    sustained_success_step: int | None
    steps_taken: int
    macro_turns: int
    failure_cluster: str
    llm_calls: int = 0
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_failure(telemetry: dict[str, Any]) -> str:
    """Classify an episode outcome into the 4 canonical failure clusters (or 'success')."""
    if telemetry.get("success_1_5x"):
        return "success"

    min_dist = float(telemetry.get("min_eef_to_target_dist", 999.0))
    if min_dist > 0.08:
        return "approach_miss"

    attempted_grasp = bool(telemetry.get("attempted_grasp", False))
    ever_grasped = bool(telemetry.get("ever_grasped", False))
    dropped_count = int(telemetry.get("dropped_count", 0))
    moved_toward_goal = bool(telemetry.get("moved_toward_goal", False))
    stage1_done = bool(telemetry.get("stage1_done", False))
    active_progress_at_end = bool(telemetry.get("active_progress_at_end", False))
    steps_taken = int(telemetry.get("steps_taken", 0))
    extended_steps = int(telemetry.get("extended_steps", 450))

    if dropped_count > 0 or (attempted_grasp and not ever_grasped and not moved_toward_goal):
        return "grasp_slip"

    if steps_taken >= extended_steps and (stage1_done or active_progress_at_end):
        return "timeout_incomplete"

    return "place_or_articulate_fail"


NOMINAL_PANDA_JOINTS = (0.0, -0.161, 0.0, -2.4446, 0.0, 2.2268, 0.7854)


def parse_macro_target(
    action_vec: Sequence[float],
    cur_eef_pos: Sequence[float],
    cur_eef_quat: Sequence[float],
    mode: str,
) -> tuple[list[float], list[float], float]:
    """Parse a 7D or 4D macro action into world-frame (target_pos, target_quat_xyzw, grip_cmd)."""
    act = [float(x) for x in action_vec]
    if len(act) not in (4, 7):
        raise ValueError(f"Macro action in mode={mode!r} must have 4 or 7 elements, got {len(act)}")

    if mode == "macro_rel_eef":
        target_pos = vec3_add(cur_eef_pos, act[:3])
        if len(act) == 4 or vec3_norm(act[3:6]) < 1e-8:
            target_quat = [float(x) for x in cur_eef_quat]
        else:
            target_quat = quat_multiply(axisangle2quat(act[3:6]), cur_eef_quat)
        return target_pos, target_quat, act[-1]

    if mode == "macro_target_eef":
        target_pos = [act[0], act[1], act[2]]
        target_quat = [float(x) for x in cur_eef_quat] if len(act) == 4 else axisangle2quat(act[3:6])
        return target_pos, target_quat, act[-1]

    raise ValueError(f"Unsupported macro control_mode: {mode!r}")


def compute_macro_osc_delta(
    cur_eef_pos: Sequence[float],
    cur_eef_quat: Sequence[float],
    target_pos: Sequence[float],
    target_quat: Sequence[float],
    grip_cmd: float,
    *,
    pos_kp: float = 12.0,
    ori_kp: float = 2.0,
) -> tuple[list[float], float, float]:
    """Compute bounded 7D OSC_POSE normalized delta via locked proportional servo law."""
    pos_err_vec = vec3_sub(target_pos, cur_eef_pos)
    pos_err = vec3_norm(pos_err_vec)
    ori_err_aa = quat_orientation_error_aa(target_quat, cur_eef_quat)
    ori_err = vec3_norm(ori_err_aa)
    u_pos = [clip(pos_kp * e, -1.0, 1.0) for e in pos_err_vec]
    u_ori = [clip(ori_kp * e, -1.0, 1.0) for e in ori_err_aa]
    g = 1.0 if float(grip_cmd) > 0.0 else -1.0
    return u_pos + u_ori + [g], pos_err, ori_err


def neutral_action_for_mode(control_mode: str, obs: dict[str, Any]) -> list[float]:
    """Return a safe zero-motion neutral hold action formatted for `control_mode`."""
    mode = normalize_control_mode(control_mode)
    if mode in ("osc_delta_eef", "macro_rel_eef"):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    if mode in ("abs_eef", "macro_target_eef"):
        eef_pos = [float(x) for x in obs.get("eef_pos", [0.0, 0.0, 0.5])]
        eef_aa = quat2axisangle(obs.get("eef_quat", [0.0, 0.0, 0.0, 1.0]))
        return eef_pos + eef_aa + [-1.0]
    if mode == "delta_joint_pos":
        return [0.0] * 7 + [-1.0]
    if mode == "abs_joint_pos":
        jpos = [float(x) for x in obs.get("joint_pos", NOMINAL_PANDA_JOINTS)]
        return jpos + [-1.0]
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


class BaseEnvAdapter:
    """Abstract simulation interface enforcing unified observation, control-mode, and step contracts."""

    def __init__(
        self,
        *,
        control_mode: str = "osc_delta_eef",
        macro_max_substeps: int = 40,
        macro_pos_tol_m: float = 0.012,
        macro_ori_tol_rad: float = 0.08,
        macro_grip_dwell_steps: int = 8,
        macro_pos_kp: float = 12.0,
        macro_ori_kp: float = 2.0,
    ) -> None:
        self.control_mode = normalize_control_mode(control_mode)
        self.macro_max_substeps = int(macro_max_substeps)
        self.macro_pos_tol_m = float(macro_pos_tol_m)
        self.macro_ori_tol_rad = float(macro_ori_tol_rad)
        self.macro_grip_dwell_steps = int(macro_grip_dwell_steps)
        self.macro_pos_kp = float(macro_pos_kp)
        self.macro_ori_kp = float(macro_ori_kp)
        self._last_obs: dict[str, Any] = {}
        self._prev_grip_cmd: float = -1.0

    def reset_episode(self, episode_spec: dict[str, Any], task_cfg: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _step_osc_delta(self, action_7d: Sequence[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        raise NotImplementedError

    def step(self, action: Sequence[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        raise NotImplementedError

    def step_macro(
        self,
        action: Sequence[float],
        *,
        max_remaining_steps: int,
        hold_target: int = 1,
        initial_consecutive_success: int = 0,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute a macro waypoint/displacement via the locked closed-loop OSC_POSE P-servo."""
        if not self._last_obs:
            raise RuntimeError("step_macro called before reset_episode")

        target_pos, target_quat, grip_cmd = parse_macro_target(
            action,
            self._last_obs["eef_pos"],
            self._last_obs["eef_quat"],
            self.control_mode,
        )
        grip_sign = 1.0 if float(grip_cmd) > 0.0 else -1.0
        grip_toggled = grip_sign != self._prev_grip_cmd
        min_steps = self.macro_grip_dwell_steps if grip_toggled else 1
        self._prev_grip_cmd = grip_sign

        limit = max(1, min(self.macro_max_substeps, int(max_remaining_steps)))
        consecutive = int(initial_consecutive_success)
        sustained_at_substep: int | None = None
        last_obs = self._last_obs
        last_reward = 0.0
        last_info: dict[str, Any] = {"is_success": False}
        steps_done = 0
        pos_err = 0.0
        ori_err = 0.0

        for sub in range(limit):
            osc_act, pos_err, ori_err = compute_macro_osc_delta(
                last_obs["eef_pos"],
                last_obs["eef_quat"],
                target_pos,
                target_quat,
                grip_sign,
                pos_kp=self.macro_pos_kp,
                ori_kp=self.macro_ori_kp,
            )
            if sub >= min_steps and pos_err <= self.macro_pos_tol_m and ori_err <= self.macro_ori_tol_rad:
                break

            last_obs, last_reward, _done, last_info = self._step_osc_delta(osc_act)
            self._last_obs = last_obs
            steps_done = sub + 1

            if last_info.get("is_success"):
                consecutive += 1
                if consecutive >= hold_target and sustained_at_substep is None:
                    sustained_at_substep = steps_done
                    break
            else:
                consecutive = 0

        if steps_done > 0:
            pos_err = vec3_dist(last_obs["eef_pos"], target_pos)
        out_info = dict(last_info)
        out_info["substeps"] = steps_done
        out_info["consecutive_success"] = consecutive
        out_info["sustained_at_substep"] = sustained_at_substep
        out_info["target_pos"] = target_pos
        out_info["pos_err"] = pos_err
        out_info["ori_err"] = ori_err
        return last_obs, last_reward, False, out_info

    def get_telemetry(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return


class MockLiberoEnvAdapter(BaseEnvAdapter):
    """Pure-Python deterministic 3D tabletop + drawer kinematic simulator for CI/Mac/dry-runs.

    Models 20 Hz 7-DoF OSC_POSE incremental motion (~1.5 cm/step), grasp close dwell physics,
    basket rim collision, drawer handle pulling, two-stage long_t0 sequencing, and all 4 OOD axes.
    """

    def __init__(
        self,
        *,
        perception_mode: str = "noisy_keypoints",
        noise_sigma_mm: float = 10.0,
        control_mode: str = "osc_delta_eef",
        macro_max_substeps: int = 40,
        macro_pos_tol_m: float = 0.012,
        macro_ori_tol_rad: float = 0.08,
        macro_grip_dwell_steps: int = 8,
        macro_pos_kp: float = 12.0,
        macro_ori_kp: float = 2.0,
    ) -> None:
        super().__init__(
            control_mode=control_mode,
            macro_max_substeps=macro_max_substeps,
            macro_pos_tol_m=macro_pos_tol_m,
            macro_ori_tol_rad=macro_ori_tol_rad,
            macro_grip_dwell_steps=macro_grip_dwell_steps,
            macro_pos_kp=macro_pos_kp,
            macro_ori_kp=macro_ori_kp,
        )
        self.perception_mode = perception_mode
        self.noise_sigma_m = float(noise_sigma_mm) / 1000.0
        self.episode_spec: dict[str, Any] = {}
        self.task_cfg: dict[str, Any] = {}
        self.step_count = 0
        self.eef_pos = [0.0, 0.0, 1.05]
        self.eef_aa = [0.0, 0.0, 0.0]
        self.joint_pos = list(NOMINAL_PANDA_JOINTS)
        self.joint_vel = [0.0] * 7
        self.gripper_opening = 0.04
        self.close_hold = 0
        self.held_obj: str | None = None
        self.objects: dict[str, list[float]] = {}
        self.goal_pos = [0.15, 0.15, 0.84]
        self.handle_pos = [0.0, 0.0, 0.88]
        self.drawer_open = 0.0
        self.drawer_target = 0.16
        self.stage1_done = False

        # Telemetry for 4-cluster attribution
        self.min_eef_dist = 999.0
        self.attempted_grasp = False
        self.ever_grasped = False
        self.dropped_count = 0
        self.moved_toward_goal = False
        self.last_goal_dist = 999.0
        self.prev_goal_dist = 999.0

    def reset_episode(self, episode_spec: dict[str, Any], task_cfg: dict[str, Any]) -> dict[str, Any]:
        self.episode_spec = dict(episode_spec)
        self.task_cfg = dict(task_cfg)
        self.step_count = 0
        seed = int(episode_spec["seed"])
        rng = random.Random(seed)
        task_key = episode_spec["task_key"]
        axis = episode_spec.get("ood_axis")
        level = int(episode_spec.get("ood_level", 0))

        # Base initial EEF pose
        self.eef_pos = [-0.08 + rng.uniform(-0.02, 0.02), -0.05 + rng.uniform(-0.02, 0.02), 1.05]
        if axis == "robot_init_state":
            amp = 0.04 * max(1, level)
            self.eef_pos[0] += rng.uniform(-amp, amp)
            self.eef_pos[1] += rng.uniform(-amp, amp)
            self.eef_pos[2] += rng.uniform(-0.03, 0.05)

        self.eef_aa = [0.0, 0.0, 0.0]
        self.joint_pos = [
            NOMINAL_PANDA_JOINTS[j] + (rng.uniform(-0.03, 0.03) if axis == "robot_init_state" else rng.uniform(-0.008, 0.008))
            for j in range(7)
        ]
        self.joint_vel = [0.0] * 7
        self.gripper_opening = 0.04
        self._prev_grip_cmd = -1.0
        self.close_hold = 0
        self.held_obj = None
        self.drawer_open = 0.0
        self.stage1_done = False

        layout_shift = [0.0, 0.0]
        if axis == "objects_layout":
            shift = 0.035 * max(1, level)
            layout_shift = [rng.uniform(-shift, shift), rng.uniform(-shift, shift)]

        if task_key == "spatial_t0":
            self.objects = {
                "black_bowl": [
                    -0.02 + rng.uniform(-0.04, 0.04) + layout_shift[0],
                    -0.12 + rng.uniform(-0.04, 0.04) + layout_shift[1],
                    0.82,
                ]
            }
            self.goal_pos = [0.16 + rng.uniform(-0.03, 0.03) + layout_shift[0] * 0.5, 0.14 + layout_shift[1] * 0.5, 0.83]
        elif task_key == "object_t0":
            self.objects = {
                "alphabet_soup": [
                    -0.06 + rng.uniform(-0.04, 0.04) + layout_shift[0],
                    -0.10 + rng.uniform(-0.04, 0.04) + layout_shift[1],
                    0.83,
                ]
            }
            self.goal_pos = [0.18 + layout_shift[0] * 0.5, 0.14 + layout_shift[1] * 0.5, 0.85]
        elif task_key == "goal_t0":
            self.objects = {}
            self.handle_pos = [
                0.08 + rng.uniform(-0.02, 0.02) + layout_shift[0],
                0.10 + rng.uniform(-0.02, 0.02) + layout_shift[1],
                0.88,
            ]
            self.goal_pos = [self.handle_pos[0] - 0.20, self.handle_pos[1], self.handle_pos[2]]
        else:  # long_t0
            self.objects = {
                "alphabet_soup": [
                    -0.09 + rng.uniform(-0.03, 0.03) + layout_shift[0],
                    -0.12 + rng.uniform(-0.03, 0.03) + layout_shift[1],
                    0.83,
                ],
                "tomato_sauce": [
                    -0.04 + rng.uniform(-0.03, 0.03) - layout_shift[0] * 0.6,
                    0.06 + rng.uniform(-0.03, 0.03) + layout_shift[1],
                    0.83,
                ],
            }
            self.goal_pos = [0.19 + layout_shift[0] * 0.4, 0.15 + layout_shift[1] * 0.4, 0.85]

        # Apply instruction paraphrasing if OOD language axis
        self._instruction = task_cfg["instruction"]
        if axis == "language_instruction":
            options = PARAPHRASED_INSTRUCTIONS.get(task_key, [self._instruction])
            self._instruction = options[seed % len(options)]

        self.min_eef_dist = 999.0
        self.attempted_grasp = False
        self.ever_grasped = False
        self.dropped_count = 0
        self.moved_toward_goal = False
        self.last_goal_dist = 999.0
        self.prev_goal_dist = 999.0

        # 10 settling steps as in real LIBERO evaluation
        for _ in range(10):
            pass

        obs = self._build_obs()
        self._last_obs = obs
        return obs

    def _active_target_pos(self) -> tuple[str, list[float]]:
        task_key = self.episode_spec["task_key"]
        if task_key == "goal_t0":
            return "drawer_handle", list(self.handle_pos)
        if task_key == "long_t0":
            if self.held_obj in self.objects:
                return self.held_obj, list(self.objects[self.held_obj])
            for name in ("alphabet_soup", "tomato_sauce"):
                if vec2_dist(self.objects[name], self.goal_pos) > 0.055 or self.objects[name][2] > 0.87:
                    return name, list(self.objects[name])
            return "tomato_sauce", list(self.objects["tomato_sauce"])
        name = next(iter(self.objects))
        return name, list(self.objects[name])

    def _build_obs(self) -> dict[str, Any]:
        quat = axisangle2quat(self.eef_aa)
        state_8d = [
            self.eef_pos[0],
            self.eef_pos[1],
            self.eef_pos[2],
            self.eef_aa[0],
            self.eef_aa[1],
            self.eef_aa[2],
            self.gripper_opening,
            -self.gripper_opening,
        ]
        seed = int(self.episode_spec["seed"])
        axis = self.episode_spec.get("ood_axis")
        sigma = self.noise_sigma_m * (2.2 if axis == "camera_visual" else 1.0)

        target_name, target_pos = self._active_target_pos()
        salt_base = self.step_count * 31
        noisy_target = [
            target_pos[i] + deterministic_gaussian_vec3(seed, salt_base + 1, sigma)[i]
            for i in range(3)
        ]
        noisy_goal = [
            self.goal_pos[i] + deterministic_gaussian_vec3(seed, salt_base + 2, sigma)[i]
            for i in range(3)
        ]

        obs: dict[str, Any] = {
            "state": state_8d,
            "eef_pos": list(self.eef_pos),
            "eef_quat": quat,
            "gripper_opening": self.gripper_opening,
            "gripper_qpos": [self.gripper_opening, -self.gripper_opening],
            "joint_pos": list(self.joint_pos),
            "joint_vel": list(self.joint_vel),
            "instruction": self._instruction,
            "task_key": self.episode_spec["task_key"],
            "step": self.step_count,
            "remaining_steps": int(self.task_cfg["extended_steps"]) - self.step_count,
        }

        if self.perception_mode == "noisy_keypoints":
            obs["keypoints"] = {
                "target_name": target_name,
                "target_xyz": noisy_target,
                "goal_xyz": noisy_goal,
                "objects": {
                    k: [
                        v[i] + deterministic_gaussian_vec3(seed, salt_base + 10 + idx, sigma)[i]
                        for i in range(3)
                    ]
                    for idx, (k, v) in enumerate(self.objects.items())
                },
                "drawer_handle_xyz": noisy_target if self.episode_spec["task_key"] == "goal_t0" else None,
            }
        else:
            # rgbd_projection mock carries lightweight 2D camera projection metadata instead of sim truth
            obs["keypoints"] = {
                "target_name": target_name,
                "target_xyz": noisy_target,
                "goal_xyz": noisy_goal,
            }
            obs["images"] = {
                "agentview": [[[0, 0, 0]]],
                "wrist": [[[0, 0, 0]]],
            }
        return obs

    def step(self, action: Sequence[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        act = [float(x) for x in action]
        if self.control_mode in MACRO_CONTROL_MODES:
            rem = max(1, int(self.task_cfg["extended_steps"]) - self.step_count)
            return self.step_macro(act, max_remaining_steps=rem)

        if self.control_mode == "osc_delta_eef":
            if len(act) != 7:
                raise ValueError(f"osc_delta_eef expects 7D action, got {len(act)}")
            return self._step_osc_delta(act, update_joints=True)

        if self.control_mode == "abs_eef":
            if len(act) != 7:
                raise ValueError(f"abs_eef expects 7D action [x,y,z,rx,ry,rz,grip], got {len(act)}")
            pos_err = vec3_sub(act[:3], self.eef_pos)
            ori_err = quat_orientation_error_aa(axisangle2quat(act[3:6]), axisangle2quat(self.eef_aa))
            u_pos = [clip(e / 0.020, -1.0, 1.0) for e in pos_err]
            u_ori = [clip(e / 0.05, -1.0, 1.0) for e in ori_err]
            return self._step_osc_delta(u_pos + u_ori + [act[6]], update_joints=True)

        if self.control_mode in JOINT_CONTROL_MODES:
            if len(act) != 8:
                raise ValueError(f"{self.control_mode} expects 8D action [q1..q7, grip], got {len(act)}")
            if self.control_mode == "delta_joint_pos":
                dq = [clip(act[j], -1.0, 1.0) * 0.05 for j in range(7)]
            else:
                dq = [clip(act[j] - self.joint_pos[j], -0.05, 0.05) for j in range(7)]
            self.joint_pos = [self.joint_pos[j] + dq[j] for j in range(7)]
            self.joint_vel = [dq[j] / 0.05 for j in range(7)]
            # Lightweight kinematic coupling to keep mock EEF responsive in joint modes
            dx_m = dq[0] * 0.25 + dq[1] * 0.35 + dq[3] * 0.20
            dy_m = dq[0] * 0.45 + dq[2] * 0.30
            dz_m = -dq[1] * 0.40 + dq[3] * 0.30 + dq[5] * 0.15
            osc_equiv = [
                clip(dx_m / 0.020, -1.0, 1.0),
                clip(dy_m / 0.020, -1.0, 1.0),
                clip(dz_m / 0.020, -1.0, 1.0),
                clip(dq[4] * 0.6 / 0.05, -1.0, 1.0),
                clip(dq[5] * 0.6 / 0.05, -1.0, 1.0),
                clip(dq[6] * 0.8 / 0.05, -1.0, 1.0),
                act[7],
            ]
            return self._step_osc_delta(osc_equiv, update_joints=False)

        raise ValueError(f"Unsupported control_mode: {self.control_mode!r}")

    def _step_osc_delta(
        self,
        action_7d: Sequence[float],
        *,
        update_joints: bool = True,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.step_count += 1
        dx = clip(float(action_7d[0]), -1.0, 1.0) * 0.020
        dy = clip(float(action_7d[1]), -1.0, 1.0) * 0.020
        dz = clip(float(action_7d[2]), -1.0, 1.0) * 0.020
        dr = [clip(float(action_7d[3 + i]), -1.0, 1.0) * 0.05 for i in range(3)]
        grip_cmd = float(action_7d[6])

        new_pos = [
            clip(self.eef_pos[0] + dx, -0.45, 0.45),
            clip(self.eef_pos[1] + dy, -0.45, 0.45),
            clip(self.eef_pos[2] + dz, 0.79, 1.30),
        ]
        self.eef_pos = new_pos
        self.eef_aa = quat2axisangle(quat_multiply(axisangle2quat(dr), axisangle2quat(self.eef_aa)))
        if update_joints:
            dq_mock = [dx * 1.2, dy * 1.2, dz * 1.2, dr[0] * 0.5, dr[1] * 0.5, dr[2] * 0.5, 0.0]
            self.joint_pos = [self.joint_pos[j] + dq_mock[j] for j in range(7)]
            self.joint_vel = [dq_mock[j] / 0.05 for j in range(7)]

        closing = grip_cmd > 0.2
        if closing:
            self.gripper_opening = max(0.0, self.gripper_opening - 0.008)
            self.close_hold += 1
            self.attempted_grasp = True
        else:
            self.gripper_opening = min(0.04, self.gripper_opening + 0.010)
            self.close_hold = 0

        task_key = self.episode_spec["task_key"]
        _, target_pos = self._active_target_pos()
        d_target = vec3_dist(self.eef_pos, target_pos)
        self.min_eef_dist = min(self.min_eef_dist, d_target)

        if task_key == "goal_t0":
            # Articulated drawer handle physics
            if closing and self.close_hold >= 5 and d_target < 0.055:
                self.ever_grasped = True
                if dx < 0.0:
                    pull = min(-dx, 0.020)
                    actual_pull = min(0.22 - self.drawer_open, pull)
                    self.drawer_open += actual_pull
                    self.handle_pos[0] -= actual_pull
                    self.moved_toward_goal = True
            elif self.held_obj is None and closing and d_target < 0.055 and self.close_hold < 5 and dx < -0.005:
                # Pulled before closing firmly -> handle slips
                self.dropped_count += 1
        else:
            # Rigid pick-and-place physics
            if self.held_obj is None:
                if closing and self.close_hold >= 5:
                    for obj_name, obj_pos in self.objects.items():
                        horiz = math.hypot(self.eef_pos[0] - obj_pos[0], self.eef_pos[1] - obj_pos[1])
                        vert = abs(self.eef_pos[2] - obj_pos[2])
                        if horiz < 0.042 and vert < 0.038:
                            self.held_obj = obj_name
                            self.ever_grasped = True
                            break
            else:
                # Carrying held object
                if not closing:
                    # Released object -> drops vertically to tabletop / basket surface
                    surface_z = 0.84 if task_key in ("object_t0", "long_t0") else 0.82
                    self.objects[self.held_obj][2] = surface_z
                    d_goal = vec2_dist(self.objects[self.held_obj], self.goal_pos)
                    if d_goal > 0.07:
                        self.dropped_count += 1
                    self.held_obj = None
                else:
                    # Check slip risk if transported fast without deep grasp dwell or low z over basket rim
                    self.objects[self.held_obj] = list(self.eef_pos)
                    cur_goal_dist = vec2_dist(self.objects[self.held_obj], self.goal_pos)
                    self.prev_goal_dist = self.last_goal_dist
                    self.last_goal_dist = cur_goal_dist
                    if cur_goal_dist < 0.16:
                        self.moved_toward_goal = True

                    if task_key in ("object_t0", "long_t0"):
                        # Basket rim collision check when horizontally crossing basket edge too low
                        if 0.065 < cur_goal_dist < 0.11 and self.eef_pos[2] < 0.89:
                            self.objects[self.held_obj][2] = 0.83
                            self.held_obj = None
                            self.dropped_count += 1

                    # Update stage1 tracking for long_t0
                    if task_key == "long_t0" and self.held_obj != "alphabet_soup":
                        soup_placed = (
                            vec2_dist(self.objects["alphabet_soup"], self.goal_pos) < 0.055
                            and self.objects["alphabet_soup"][2] <= 0.87
                        )
                        if soup_placed:
                            self.stage1_done = True

        success_now = self._check_success()
        obs = self._build_obs()
        self._last_obs = obs
        reward = 1.0 if success_now else 0.0
        return obs, reward, False, {"is_success": success_now}

    def _check_success(self) -> bool:
        task_key = self.episode_spec["task_key"]
        if task_key == "goal_t0":
            return self.drawer_open >= 0.145
        if task_key == "long_t0":
            return all(
                vec2_dist(pos, self.goal_pos) < 0.055 and pos[2] <= 0.87 and self.held_obj != name
                for name, pos in self.objects.items()
            )
        name = next(iter(self.objects))
        pos = self.objects[name]
        return vec2_dist(pos, self.goal_pos) < 0.050 and pos[2] <= 0.87 and self.held_obj is None

    def get_telemetry(self) -> dict[str, Any]:
        return {
            "min_eef_to_target_dist": self.min_eef_dist,
            "attempted_grasp": self.attempted_grasp,
            "ever_grasped": self.ever_grasped,
            "dropped_count": self.dropped_count,
            "moved_toward_goal": self.moved_toward_goal,
            "stage1_done": self.stage1_done,
            "active_progress_at_end": self.last_goal_dist < self.prev_goal_dist or self.drawer_open > 0.04,
        }


DEFAULT_OOD_AXIS_MAPPING = {
    "objects_layout": ["Objects Layout"],
    "robot_init_state": ["Robot Initial States"],
    "language_instruction": ["Language Instructions"],
    "camera_visual": ["Camera Viewpoints", "Light Conditions", "Background Textures", "Sensor Noise"],
}


@contextmanager
def _allow_unsafe_torch_load() -> Any:
    """Temporarily default torch.load(..., weights_only=False) for PyTorch 2.6+ .pruned_init files."""
    import torch  # type: ignore[import-not-found]

    orig_load = torch.load

    def _patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = _patched_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = orig_load  # type: ignore[assignment]


def _torch_load_unsafe(path: str) -> Any:
    import torch  # type: ignore[import-not-found]

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_uint8_hwc(arr: Any) -> Any:
    img = np.ascontiguousarray(arr[::-1, ::-1])
    if img.dtype != np.uint8:
        if float(np.max(img)) <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class LiberoPlusEnvAdapter(BaseEnvAdapter):
    """Real sylvestf/LIBERO-plus (MuJoCo/robosuite) adapter with strict observation contract."""

    def __init__(
        self,
        *,
        perception_mode: str = "rgbd_projection",
        noise_sigma_mm: float = 10.0,
        image_size: int = 256,
        ood_axis_mapping: dict[str, list[str]] | None = None,
        control_mode: str = "osc_delta_eef",
        macro_max_substeps: int = 40,
        macro_pos_tol_m: float = 0.012,
        macro_ori_tol_rad: float = 0.08,
        macro_grip_dwell_steps: int = 8,
        macro_pos_kp: float = 12.0,
        macro_ori_kp: float = 2.0,
    ) -> None:
        super().__init__(
            control_mode=control_mode,
            macro_max_substeps=macro_max_substeps,
            macro_pos_tol_m=macro_pos_tol_m,
            macro_ori_tol_rad=macro_ori_tol_rad,
            macro_grip_dwell_steps=macro_grip_dwell_steps,
            macro_pos_kp=macro_pos_kp,
            macro_ori_kp=macro_ori_kp,
        )
        try:
            from libero.libero import benchmark, get_libero_path  # type: ignore[import-not-found]
            from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]
            from robosuite.utils.camera_utils import (  # type: ignore[import-not-found]
                get_camera_extrinsic_matrix,
                get_camera_intrinsic_matrix,
                get_real_depth_map,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LIBERO-plus is not installed in this Python environment. "
                "Install sylvestf/LIBERO-plus or pass --env-backend mock for local dry-runs."
            ) from exc

        self._benchmark = benchmark
        self._get_libero_path = get_libero_path
        self._OffScreenRenderEnv = OffScreenRenderEnv
        self._get_camera_intrinsic_matrix = get_camera_intrinsic_matrix
        self._get_camera_extrinsic_matrix = get_camera_extrinsic_matrix
        self._get_real_depth_map = get_real_depth_map
        self.perception_mode = perception_mode
        self.noise_sigma_m = float(noise_sigma_mm) / 1000.0
        self.image_size = int(image_size)
        self.ood_axis_mapping = ood_axis_mapping or DEFAULT_OOD_AXIS_MAPPING

        self._suite_cache: dict[str, Any] = {}
        self._task_classification: dict[str, list[dict[str, Any]]] | None = None
        self._nominal_camera_cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        self._current_cam_K: dict[str, Any] = {}
        self._current_cam_T: dict[str, Any] = {}
        self._env: Any = None
        self._raw_obs: dict[str, Any] = {}
        self._episode_spec: dict[str, Any] = {}
        self._task_cfg: dict[str, Any] = {}
        self._instruction = ""
        self._step_count = 0
        self._min_eef_dist = 999.0
        self._attempted_grasp = False
        self._ever_grasped = False
        self._dropped_count = 0

    def _get_suite(self, suite_name: str) -> Any:
        if suite_name not in self._suite_cache:
            benchmark_dict = self._benchmark.get_benchmark_dict()
            self._suite_cache[suite_name] = benchmark_dict[suite_name]()
        return self._suite_cache[suite_name]

    def _get_task_classification(self) -> dict[str, list[dict[str, Any]]]:
        if self._task_classification is None:
            tc_path = Path(self._get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
            self._task_classification = json.loads(tc_path.read_text(encoding="utf-8"))
        return self._task_classification

    def _resolve_ood_variant(self, episode_spec: dict[str, Any], task_cfg: dict[str, Any]) -> tuple[Any, int, dict[str, Any]]:
        suite_name = episode_spec["suite"]
        task_suite = self._get_suite(suite_name)
        tc = self._get_task_classification()
        axis = str(episode_spec["ood_axis"])
        clean_base_name = str(episode_spec.get("clean_base_name") or task_cfg["clean_base_name"])
        target_cats = set(self.ood_axis_mapping[axis])
        slot = int(episode_spec.get("variant_slot", 0))

        candidates = [
            item
            for item in tc.get(suite_name, [])
            if item.get("category") in target_cats
            and str(item.get("name", "")).startswith(clean_base_name)
            and int(item.get("difficulty_level", 0)) in (2, 3, 4)
        ]
        if not candidates:
            candidates = [
                item
                for item in tc.get(suite_name, [])
                if item.get("category") in target_cats and str(item.get("name", "")).startswith(clean_base_name)
            ]
        if not candidates:
            raise KeyError(
                f"No LIBERO-plus OOD variants found for suite={suite_name!r}, "
                f"clean_base_name={clean_base_name!r}, axis={axis!r}"
            )
        candidates.sort(key=lambda item: int(item["id"]))
        chosen = candidates[slot % len(candidates)]
        tid = int(chosen["id"]) - 1
        return task_suite, tid, chosen

    def reset_episode(self, episode_spec: dict[str, Any], task_cfg: dict[str, Any]) -> dict[str, Any]:
        self.close()
        self._episode_spec = dict(episode_spec)
        self._task_cfg = dict(task_cfg)
        self._step_count = 0
        self._min_eef_dist = 999.0
        self._attempted_grasp = False
        self._ever_grasped = False
        self._dropped_count = 0

        suite_name = str(episode_spec["suite"])
        clean_base_name = str(episode_spec.get("clean_base_name") or task_cfg["clean_base_name"])
        ood_axis = episode_spec.get("ood_axis")
        robosuite_ctrl = "JOINT_POSITION" if self.control_mode in JOINT_CONTROL_MODES else "OSC_POSE"

        if ood_axis is None:
            # Clean ID / Dev / Burned splits load the clean .bddl and .pruned_init directly
            bddl_path = os.path.join(self._get_libero_path("bddl_files"), suite_name, f"{clean_base_name}.bddl")
            init_path = os.path.join(self._get_libero_path("init_states"), suite_name, f"{clean_base_name}.pruned_init")
            self._env = self._OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                controller=robosuite_ctrl,
                camera_heights=self.image_size,
                camera_widths=self.image_size,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_depths=True,
                horizon=1000,
            )
            self._env.seed(int(episode_spec["seed"]))
            self._raw_obs = self._env.reset()

            init_id = episode_spec.get("init_state_id")
            if init_id is not None:
                clean_inits = _torch_load_unsafe(init_path)
                self._raw_obs = self._env.set_init_state(clean_inits[int(init_id)])

            self._instruction = str(task_cfg["instruction"])
        else:
            # OOD variants resolve via LIBERO-plus task_classification.json
            task_suite, tid, _ = self._resolve_ood_variant(episode_spec, task_cfg)
            task = task_suite.get_task(tid)
            bddl_path = task_suite.get_task_bddl_file_path(tid)
            with _allow_unsafe_torch_load():
                inits = task_suite.get_task_init_states(tid)

            self._env = self._OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                controller=robosuite_ctrl,
                camera_heights=self.image_size,
                camera_widths=self.image_size,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_depths=True,
                horizon=1000,
            )
            self._env.seed(int(episode_spec["seed"]))
            self._raw_obs = self._env.reset()

            if ood_axis == "robot_init_state":
                # Preserve perturbed robot joint/arm pose created by reset() while loading object states
                robot_qpos = self._env.sim.data.qpos[:9].copy()
                self._env.set_init_state(inits[0])
                self._env.sim.data.qpos[:9] = robot_qpos
                self._env.sim.data.qvel[:9] = 0.0
                self._env.sim.forward()
                self._raw_obs = self._env.env._get_observations()
            else:
                self._raw_obs = self._env.set_init_state(inits[0])

            # Only expose LIBERO-plus rewritten task.language on language_instruction OOD axis
            if ood_axis == "language_instruction":
                self._instruction = str(task.language)
            else:
                self._instruction = str(task_cfg["instruction"])

        # Mandatory 10 neutral settling steps after reset / set_init_state
        dummy_action = [0.0] * 7 + [-1.0] if robosuite_ctrl == "JOINT_POSITION" else [0.0] * 6 + [-1.0]
        for _ in range(10):
            self._raw_obs, _, _, _ = self._env.step(dummy_action)

        # Switch OSC controller to absolute EEF pose ONLY after neutral zero-delta settling
        if self.control_mode == "abs_eef":
            self._env.env.robots[0].controller.use_delta = False

        self._prev_grip_cmd = -1.0
        self._ensure_nominal_cameras(suite_name, clean_base_name, ood_axis)
        obs = self._sanitize_obs(self._raw_obs)
        self._last_obs = obs
        return obs

    def _compute_rot180_cameras(self, env_obj: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute camera intrinsics & extrinsics aligned to [::-1, ::-1] rotated images."""
        intrinsics: dict[str, Any] = {}
        extrinsics: dict[str, Any] = {}
        for cam in ("agentview", "robot0_eye_in_hand"):
            K = self._get_camera_intrinsic_matrix(env_obj.sim, cam, self.image_size, self.image_size).copy()
            T_wc = self._get_camera_extrinsic_matrix(env_obj.sim, cam).copy()
            # OpenGL->OpenCV flips rows; [::-1, ::-1] flips columns relative to OpenCV -> mirror u & X_cam
            K[0, 2] = (self.image_size - 1.0) - K[0, 2]
            T_wc[:3, 0] *= -1.0
            intrinsics[cam] = K.tolist()
            extrinsics[cam] = T_wc.tolist()
        return intrinsics, extrinsics

    def _ensure_nominal_cameras(self, suite_name: str, clean_base_name: str, ood_axis: str | None) -> None:
        key = (suite_name, clean_base_name)
        if key not in self._nominal_camera_cache:
            if ood_axis is None or ood_axis != "camera_visual":
                self._nominal_camera_cache[key] = self._compute_rot180_cameras(self._env)
            else:
                # Calibrate nominal cameras from the unperturbed BDDL so camera_visual OOD never leaks shifted extrinsics
                bddl_path = os.path.join(self._get_libero_path("bddl_files"), suite_name, f"{clean_base_name}.bddl")
                tmp_env = self._OffScreenRenderEnv(
                    bddl_file_name=bddl_path,
                    camera_heights=self.image_size,
                    camera_widths=self.image_size,
                    camera_names=["agentview", "robot0_eye_in_hand"],
                    horizon=100,
                )
                try:
                    tmp_env.reset()
                    self._nominal_camera_cache[key] = self._compute_rot180_cameras(tmp_env)
                finally:
                    tmp_env.close()
        self._current_cam_K, self._current_cam_T = self._nominal_camera_cache[key]

    def _sanitize_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        eef_pos = [float(x) for x in raw["robot0_eef_pos"]]
        eef_quat = [float(x) for x in raw["robot0_eef_quat"]]
        eef_aa = quat2axisangle(eef_quat)
        grip = [float(x) for x in raw["robot0_gripper_qpos"][:2]]
        joint_pos = [float(x) for x in raw["robot0_joint_pos"]]
        joint_vel = [float(x) for x in raw["robot0_joint_vel"]]
        state_8d = eef_pos + eef_aa + grip

        # Mandatory 180-degree camera flip matching LeRobot/LIBERO evaluation protocol
        agent_img = _to_uint8_hwc(raw["agentview_image"])
        wrist_img = _to_uint8_hwc(raw["robot0_eye_in_hand_image"])

        obs: dict[str, Any] = {
            "state": state_8d,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "gripper_opening": float(grip[0]),
            "gripper_qpos": grip,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "instruction": self._instruction,
            "task_key": self._episode_spec["task_key"],
            "step": self._step_count,
            "remaining_steps": int(self._task_cfg["extended_steps"]) - self._step_count,
            "images": {
                "agentview": agent_img,
                "wrist": wrist_img,
            },
        }
        # Never expose raw sim.data / object-state to policies
        if self.perception_mode == "rgbd_projection":
            if "agentview_depth" in raw:
                agent_d_m = self._get_real_depth_map(self._env.sim, raw["agentview_depth"])
                wrist_d_m = self._get_real_depth_map(self._env.sim, raw["robot0_eye_in_hand_depth"])
                obs["depth"] = {
                    "agentview": np.ascontiguousarray(agent_d_m[::-1, ::-1]).astype(np.float32),
                    "wrist": np.ascontiguousarray(wrist_d_m[::-1, ::-1]).astype(np.float32),
                }
            if self._current_cam_K:
                obs["camera_intrinsics"] = self._current_cam_K
                obs["camera_extrinsics"] = self._current_cam_T
        elif self.perception_mode == "noisy_keypoints" and "object-state" in raw and len(raw["object-state"]) >= 3:
            seed = int(self._episode_spec["seed"])
            noise = deterministic_gaussian_vec3(seed, self._step_count * 17 + 5, self.noise_sigma_m)
            obj_vec = [float(x) for x in raw["object-state"][:3]]
            obs["keypoints"] = {
                "target_xyz": [obj_vec[i] + noise[i] for i in range(3)],
            }
        return obs

    def _step_raw(self, raw_action: list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._step_count += 1
        if float(raw_action[-1]) > 0.2:
            self._attempted_grasp = True
        self._raw_obs, reward, done, info = self._env.step(list(raw_action))
        if "object-state" in self._raw_obs and len(self._raw_obs["object-state"]) >= 3 and "robot0_eef_pos" in self._raw_obs:
            d = vec3_dist(self._raw_obs["robot0_eef_pos"][:3], self._raw_obs["object-state"][:3])
            self._min_eef_dist = min(self._min_eef_dist, d)
            if d < 0.06 and float(raw_action[-1]) > 0.2:
                self._ever_grasped = True
        is_succ = bool(done or self._env.check_success())
        info = dict(info or {})
        info["is_success"] = is_succ
        obs = self._sanitize_obs(self._raw_obs)
        self._last_obs = obs
        return obs, float(reward), False, info

    def _step_osc_delta(self, action_7d: Sequence[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        clipped = [clip(float(action_7d[i]), -1.0, 1.0) for i in range(6)] + [float(action_7d[6])]
        return self._step_raw(clipped)

    def step(self, action: Sequence[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        act = [float(x) for x in action]
        if self.control_mode in MACRO_CONTROL_MODES:
            rem = max(1, int(self._task_cfg["extended_steps"]) - self._step_count)
            return self.step_macro(act, max_remaining_steps=rem)

        if self.control_mode == "osc_delta_eef":
            if len(act) != 7:
                raise ValueError(f"osc_delta_eef expects 7D action, got {len(act)}")
            return self._step_osc_delta(act)

        if self.control_mode == "abs_eef":
            if len(act) != 7:
                raise ValueError(f"abs_eef expects 7D action [x,y,z,rx,ry,rz,grip], got {len(act)}")
            return self._step_raw(act)

        if self.control_mode == "delta_joint_pos":
            if len(act) != 8:
                raise ValueError(f"delta_joint_pos expects 8D action [dq1..dq7,grip], got {len(act)}")
            return self._step_raw([clip(act[i], -1.0, 1.0) for i in range(7)] + [act[7]])

        if self.control_mode == "abs_joint_pos":
            if len(act) != 8:
                raise ValueError(f"abs_joint_pos expects 8D action [q1..q7,grip], got {len(act)}")
            cur_q = self._raw_obs["robot0_joint_pos"]
            dq = [clip((act[i] - float(cur_q[i])) / 0.05, -1.0, 1.0) for i in range(7)] + [act[7]]
            return self._step_raw(dq)

        raise ValueError(f"Unsupported control_mode: {self.control_mode!r}")

    def get_telemetry(self) -> dict[str, Any]:
        return {
            "min_eef_to_target_dist": self._min_eef_dist if self._min_eef_dist < 100 else 0.05,
            "attempted_grasp": self._attempted_grasp,
            "ever_grasped": self._ever_grasped,
            "dropped_count": self._dropped_count,
            "moved_toward_goal": False,
            "stage1_done": False,
            "active_progress_at_end": False,
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def create_env_adapter(
    backend: str,
    *,
    perception_mode: str = "noisy_keypoints",
    noise_sigma_mm: float = 10.0,
    control_mode: str = "osc_delta_eef",
) -> BaseEnvAdapter:
    """Factory returning either the real LIBERO-Plus adapter or pure-Python mock simulator."""
    mode = normalize_control_mode(control_mode)
    if backend == "mock":
        return MockLiberoEnvAdapter(
            perception_mode=perception_mode,
            noise_sigma_mm=noise_sigma_mm,
            control_mode=mode,
        )
    if backend == "libero":
        return LiberoPlusEnvAdapter(
            perception_mode=perception_mode,
            noise_sigma_mm=noise_sigma_mm,
            control_mode=mode,
        )
    raise ValueError(f"Unknown env backend: {backend!r} (expected 'mock' or 'libero')")


def rollout_episode(
    env: BaseEnvAdapter,
    policy: BasePolicy,
    episode_spec: dict[str, Any],
    task_cfg: dict[str, Any],
    *,
    ledger: CostLedger | None = None,
) -> EpisodeResult:
    """Execute one unified dual-horizon rollout matching official LIBERO success verification."""
    standard_steps = int(task_cfg["standard_steps"])
    extended_steps = int(task_cfg["extended_steps"])
    hold_target = int(task_cfg.get("hold_success_steps", 1))
    max_macro_turns = int(task_cfg.get("max_macro_turns", 15))
    mode = normalize_control_mode(getattr(env, "control_mode", getattr(policy, "control_mode", "osc_delta_eef")))

    obs = env.reset_episode(episode_spec, task_cfg)
    policy.reset(
        {
            "episode_id": episode_spec["episode_id"],
            "task_key": episode_spec["task_key"],
            "instruction": obs.get("instruction", task_cfg["instruction"]),
            "standard_steps": standard_steps,
            "extended_steps": extended_steps,
            "max_macro_turns": max_macro_turns,
            "control_mode": mode,
        }
    )

    steps_taken = 0
    macro_turns = 0
    episode_llm_calls = 0
    episode_tokens = 0
    consecutive_success = 0
    sustained_success_step: int | None = None
    action_queue: list[list[float]] = []

    while steps_taken < extended_steps:
        if not action_queue:
            if policy.family == "agentic" and macro_turns >= max_macro_turns:
                action_queue = [neutral_action_for_mode(mode, obs)]
            else:
                raw_action, call_cost = policy.act(obs)
                if ledger is not None:
                    ledger.add_call_cost(call_cost)
                episode_llm_calls += int((call_cost or {}).get("llm_total_calls", 0))
                episode_tokens += (
                    int((call_cost or {}).get("prompt_tokens", 0))
                    + int((call_cost or {}).get("completion_tokens", 0))
                    + int((call_cost or {}).get("reasoning_tokens", 0))
                )
                macro_inc = int((call_cost or {}).get("macro_turn_increment", 1 if policy.family == "agentic" else 0))
                macro_turns += macro_inc

                if not raw_action:
                    action_queue = [neutral_action_for_mode(mode, obs)]
                elif isinstance(raw_action[0], (list, tuple)):
                    action_queue = [[float(x) for x in step_a] for step_a in raw_action]
                else:
                    action_queue = [[float(x) for x in raw_action]]

        act_vec = action_queue.pop(0)
        if mode in MACRO_CONTROL_MODES:
            obs, _reward, _done, info = env.step_macro(
                act_vec,
                max_remaining_steps=extended_steps - steps_taken,
                hold_target=hold_target,
                initial_consecutive_success=consecutive_success,
            )
            substeps = int(info.get("substeps", 1))
            consecutive_success = int(info.get("consecutive_success", 0))
            sustained_offset = info.get("sustained_at_substep")
            steps_taken += substeps
            if sustained_offset is not None and sustained_success_step is None:
                sustained_success_step = (steps_taken - substeps) + int(sustained_offset)
                break
        else:
            obs, _reward, _done, info = env.step(act_vec)
            steps_taken += 1

            if info.get("is_success"):
                consecutive_success += 1
                if consecutive_success >= hold_target and sustained_success_step is None:
                    sustained_success_step = steps_taken
                    break
            else:
                consecutive_success = 0

    success_1_5x = sustained_success_step is not None and sustained_success_step <= extended_steps
    success_1_0x = sustained_success_step is not None and sustained_success_step <= standard_steps

    telemetry = env.get_telemetry()
    telemetry.update(
        {
            "success_1_5x": success_1_5x,
            "steps_taken": steps_taken,
            "extended_steps": extended_steps,
        }
    )
    cluster = classify_failure(telemetry)

    if ledger is not None:
        ledger.online_episodes += 1
        ledger.online_env_steps += steps_taken

    return EpisodeResult(
        episode_id=episode_spec["episode_id"],
        task_key=episode_spec["task_key"],
        split=episode_spec["split"],
        ood_axis=episode_spec.get("ood_axis"),
        success_1_0x=success_1_0x,
        success_1_5x=success_1_5x,
        sustained_success_step=sustained_success_step,
        steps_taken=steps_taken,
        macro_turns=macro_turns,
        failure_cluster=cluster,
        llm_calls=episode_llm_calls,
        tokens=episode_tokens,
    )
