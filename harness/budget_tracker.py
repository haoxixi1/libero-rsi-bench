"""Online episode budget enforcement, sealed-split protection, and 4D cost ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BudgetExceededError(RuntimeError):
    """Raised when online training/RSI exceeds the allowed Dev episode budget."""


class SealedAccessViolationError(PermissionError):
    """Raised when a training/optimization loop tries to read sealed test splits."""


@dataclass
class CostLedger:
    """Four-dimensional lifecycle cost ledger matching the unified benchmark spec."""

    demo_count: int = 0
    online_episodes: int = 0
    online_env_steps: int = 0
    gpu_hours: float = 0.0
    llm_unique_calls: int = 0
    llm_total_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    eval_wall_sec: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

    def add_call_cost(self, call_cost: dict[str, Any] | None) -> None:
        """Accumulate per-step LLM/API call metrics returned by a policy."""
        if not call_cost:
            return
        self.llm_unique_calls += int(call_cost.get("llm_unique_calls", 0))
        self.llm_total_calls += int(call_cost.get("llm_total_calls", 0))
        self.prompt_tokens += int(call_cost.get("prompt_tokens", 0))
        self.completion_tokens += int(call_cost.get("completion_tokens", 0))
        self.reasoning_tokens += int(call_cost.get("reasoning_tokens", 0))

    def merge(self, other: "CostLedger") -> None:
        """Add counts from another CostLedger into this ledger."""
        self.demo_count = max(self.demo_count, other.demo_count)
        self.online_episodes += other.online_episodes
        self.online_env_steps += other.online_env_steps
        self.gpu_hours += other.gpu_hours
        self.llm_unique_calls += other.llm_unique_calls
        self.llm_total_calls += other.llm_total_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.eval_wall_sec += other.eval_wall_sec

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


class OnlineBudgetGuard:
    """Enforces the 2,000-episode Dev budget and audits sealed test split access."""

    def __init__(
        self,
        max_online_episodes: int = 2000,
        sealed_log_path: Path | str | None = None,
    ) -> None:
        self.max_online_episodes = int(max_online_episodes)
        self.used_online_episodes = 0
        self.used_online_steps = 0
        self.sealed_log_path = (
            Path(sealed_log_path)
            if sealed_log_path is not None
            else Path(__file__).resolve().parent.parent / "runs" / "sealed_reads.log"
        )

    def check_split_access(
        self,
        split_name: str,
        *,
        allow_unseal: bool = False,
        policy_id: str = "unknown",
        sealed_sha256: str = "",
        num_episodes: int = 0,
    ) -> None:
        """Block unauthorized sealed reads and write audit logs on valid unseal."""
        if split_name in ("sealed_id", "sealed_ood"):
            if not allow_unseal:
                raise SealedAccessViolationError(
                    f"Access to '{split_name}' denied without explicit allow_unseal=True (--confirm-unseal)."
                )
            self._append_sealed_audit_log(
                policy_id=policy_id,
                split_name=split_name,
                sealed_sha256=sealed_sha256,
                num_episodes=num_episodes,
            )

    def consume_online_episode(self, steps: int = 0, count: int = 1) -> None:
        """Consume online Dev interaction episodes and fail hard if budget is exceeded."""
        next_episodes = self.used_online_episodes + int(count)
        if next_episodes > self.max_online_episodes:
            raise BudgetExceededError(
                f"Online Dev episode budget exceeded: attempted {next_episodes} > cap {self.max_online_episodes}"
            )
        self.used_online_episodes = next_episodes
        self.used_online_steps += int(steps)

    @property
    def remaining_episodes(self) -> int:
        return max(0, self.max_online_episodes - self.used_online_episodes)

    def _append_sealed_audit_log(
        self,
        *,
        policy_id: str,
        split_name: str,
        sealed_sha256: str,
        num_episodes: int,
    ) -> None:
        self.sealed_log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts} | policy={policy_id} | split={split_name} | sha256={sealed_sha256} | episodes={num_episodes}\n"
        with self.sealed_log_path.open("a", encoding="utf-8") as f:
            f.write(line)
