"""Minimal template showing how RL / RSI teammates interact with the Dev split safely."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.budget_tracker import BudgetExceededError
from harness.dev_env import DevInteractionEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a budget-guarded Dev interaction demo.")
    parser.add_argument("--task-key", default=None, help="Optional single task (e.g. spatial_t0)")
    parser.add_argument("--env-backend", choices=("mock", "libero"), default="mock")
    parser.add_argument("--max-online-episodes", type=int, default=3)
    args = parser.parse_args()

    env = DevInteractionEnv(
        task_key=args.task_key,
        env_backend=args.env_backend,
        max_online_episodes=args.max_online_episodes,
    )
    try:
        for ep_idx in range(args.max_online_episodes):
            obs, info = env.reset()
            for _ in range(5):
                obs, reward, terminated, truncated, step_info = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                if terminated or truncated:
                    break
            print(
                f"Episode {ep_idx + 1}: id={info['episode_id']} task={info['task_key']} "
                f"remaining_budget={info['remaining_budget']} instr={obs['instruction']!r}"
            )
        # Attempting one extra reset demonstrates automatic BudgetExceededError protection
        try:
            env.reset()
        except BudgetExceededError as exc:
            print(f"Budget guard verified: {exc}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
