"""Minimal template showing Code-as-Policies (imperative Python script + loops) via HTTP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.policy_protocol import ImperativePolicyBase, PolicyHTTPServer, RobotCodeAPI


class MinimalCodeAsPolicy(ImperativePolicyBase):
    """Execute imperative LLM-synthesized Python code directly via `RobotCodeAPI`."""

    policy_id = "example_code_policy"
    family = "agentic"
    control_mode = "macro_target_eef"

    def run_episode(self, api: RobotCodeAPI, task_meta: dict[str, Any]) -> None:
        # Replace this static snippet with your VLM/LLM code synthesizer on `task_meta["instruction"]`.
        api.record_llm_cost(prompt_tokens=0, completion_tokens=0, unique_calls=0, total_calls=0)
        generated_code = """
obs = api.get_obs()
cur = obs["eef_pos"]
obs = api.move_delta(0.0, 0.0, -0.05)
if obs["gripper_opening"] > 0.02:
    obs = api.close_gripper()
obs = api.move_to(cur[0], cur[1], cur[2], gripper=-1.0)
"""
        exec(generated_code, {"api": api, "task_meta": task_meta})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a minimal Code-as-Policies HTTP server.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    policy = MinimalCodeAsPolicy()
    server = PolicyHTTPServer(policy, host=args.host, port=args.port)
    print(f"Serving {policy.policy_id} (control_mode={policy.control_mode}) at {server.url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
