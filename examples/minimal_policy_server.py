"""Minimal method-agnostic template: expose ANY IL/RL/Agentic/RSI policy via HTTP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.env_adapter import neutral_action_for_mode
from harness.policy_protocol import (
    BasePolicy,
    CONTROL_MODE_ALIASES,
    PolicyHTTPServer,
    VALID_CONTROL_MODES,
    normalize_control_mode,
)


class MinimalTemplatePolicy(BasePolicy):
    """Replace `reset` and `act` with your own model, RL agent, VLM planner, or RSI policy."""

    def __init__(
        self,
        family: str = "il",
        control_mode: str = "osc_delta_eef",
        chunk_size: int = 10,
    ) -> None:
        self.policy_id = f"example_{family}_policy"
        self.family = family
        self.control_mode = normalize_control_mode(control_mode)
        self.chunk_size = int(chunk_size)
        self._last_instruction = ""

    def reset(self, task_meta: dict[str, Any]) -> None:
        self._last_instruction = str(task_meta.get("instruction", ""))

    def act(self, obs_dict: dict[str, Any]) -> tuple[list[float] | list[list[float]], dict[str, Any]]:
        # Available keys in `obs_dict`:
        #   - obs_dict["images"]["agentview"]: uint8 (256, 256, 3), already rotated 180 deg
        #   - obs_dict["images"]["wrist"]:     uint8 (256, 256, 3), already rotated 180 deg
        #   - obs_dict["depth"]["agentview"]:  float32 (256, 256, 1) metric meters (in rgbd_projection mode)
        #   - obs_dict["camera_intrinsics"] / ["camera_extrinsics"]: calibrated for rotated images
        #   - obs_dict["state"]:               list[8] (eef_pos[3] + axisangle[3] + gripper[2])
        #   - obs_dict["joint_pos"] / ["joint_vel"]: list[7] (Panda 7-DoF angles rad & rad/s)
        #   - obs_dict["gripper_qpos"]:        list[2]
        #   - obs_dict["instruction"]:         str
        #   - obs_dict["step"], obs_dict["remaining_steps"]
        #
        # Action formats by `self.control_mode` (declared via GET /health):
        #   - osc_delta_eef (7D):    [dx, dy, dz, drx, dry, drz, grip] in [-1, 1]
        #   - abs_eef (7D):          [x, y, z, rx, ry, rz, grip] world-frame pose (m, rad)
        #   - abs_joint_pos (8D):    [q1..q7, grip] absolute joint angles (rad)
        #   - delta_joint_pos (8D):  [dq1..dq7, grip] normalized [-1, 1] joint deltas
        #   - macro_rel_eef (4D/7D): [dx_m, dy_m, dz_m, (drx, dry, drz,) grip] closed-loop P-servo
        #   - macro_target_eef (4D/7D): [x, y, z, (rx, ry, rz,) grip] closed-loop P-servo
        _ = obs_dict.get("images")
        single_step = neutral_action_for_mode(self.control_mode, obs_dict)
        action = [list(single_step) for _ in range(self.chunk_size)] if self.chunk_size > 1 else single_step

        # Optional cost metadata (important for Agentic / RSI LLM call & token tracking)
        call_cost = {
            "macro_turn_increment": 1 if self.family == "agentic" else 0,
            "llm_unique_calls": 0,
            "llm_total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }
        return action, call_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a minimal template policy over HTTP.")
    parser.add_argument("--family", choices=("il", "rl", "agentic", "rsi"), default="il")
    parser.add_argument(
        "--control-mode",
        choices=VALID_CONTROL_MODES + tuple(CONTROL_MODE_ALIASES),
        default="osc_delta_eef",
    )
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    policy = MinimalTemplatePolicy(
        family=args.family,
        control_mode=args.control_mode,
        chunk_size=args.chunk_size,
    )
    server = PolicyHTTPServer(policy, host=args.host, port=args.port)
    print(f"Serving {policy.policy_id} (control_mode={policy.control_mode}, chunk_size={args.chunk_size}) at {server.url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
