"""Expose a trained ILChunkingPolicy over HTTP (optional decoupled server mode)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines.il_policy import ILChunkingPolicy
from harness.policy_protocol import PolicyHTTPServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a trained ILChunkingPolicy over HTTP.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--no-temporal-ensemble", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    policy = ILChunkingPolicy(
        checkpoint_path=args.checkpoint,
        exec_horizon=args.exec_horizon,
        temporal_ensemble=not args.no_temporal_ensemble,
    )
    server = PolicyHTTPServer(policy, host=args.host, port=args.port)
    print(f"Serving IL policy '{policy.policy_id}' (mode={policy.control_mode}) at {server.url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
