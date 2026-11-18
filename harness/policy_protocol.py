"""Decoupled cross-environment Policy protocol (BasePolicy + JSON HTTP Server/Client)."""

from __future__ import annotations

import base64
import json
import queue
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from harness.math_kinematics import (
    axisangle2quat,
    quat2axisangle,
    quat_multiply,
    vec3_add,
    vec3_sub,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


VALID_FAMILIES = ("il", "rl", "agentic", "rsi")
VALID_CONTROL_MODES = (
    "osc_delta_eef",
    "abs_eef",
    "abs_joint_pos",
    "delta_joint_pos",
    "macro_rel_eef",
    "macro_target_eef",
)
CONTROL_MODE_ALIASES = {
    "delta_eef": "osc_delta_eef",
    "target_eef": "macro_target_eef",
}
MACRO_CONTROL_MODES = ("macro_rel_eef", "macro_target_eef")
JOINT_CONTROL_MODES = ("abs_joint_pos", "delta_joint_pos")


def normalize_control_mode(mode: str | None) -> str:
    """Normalize control mode strings and aliases into canonical mode identifiers."""
    if mode is None:
        return "osc_delta_eef"
    canonical = CONTROL_MODE_ALIASES.get(str(mode), str(mode))
    if canonical not in VALID_CONTROL_MODES:
        raise ValueError(
            f"Unknown control_mode {mode!r}; expected one of {VALID_CONTROL_MODES} "
            f"(aliases: {tuple(CONTROL_MODE_ALIASES)})"
        )
    return canonical


def _encode_value(val: Any) -> Any:
    """Serialize numpy arrays or nested Python containers into JSON-safe structures."""
    if np is not None and isinstance(val, np.ndarray):
        raw = base64.b64encode(np.ascontiguousarray(val).tobytes()).decode("ascii")
        return {
            "__ndarray__": True,
            "dtype": str(val.dtype),
            "shape": list(val.shape),
            "data_b64": raw,
        }
    if isinstance(val, dict):
        return {str(k): _encode_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_encode_value(v) for v in val]
    if hasattr(val, "item") and callable(val.item):
        return val.item()
    return val


def _decode_value(val: Any) -> Any:
    """Restore base64-encoded arrays (into numpy.ndarray if available, else nested lists)."""
    if isinstance(val, dict):
        if val.get("__ndarray__") is True:
            raw = base64.b64decode(val["data_b64"].encode("ascii"))
            shape = tuple(int(s) for s in val["shape"])
            dtype = str(val["dtype"])
            if np is not None:
                return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
            # Pure-Python fallback for uint8 / float32 / float64 buffers
            return _bytes_to_nested_list(raw, dtype, shape)
        return {k: _decode_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_decode_value(v) for v in val]
    return val


def _bytes_to_nested_list(raw: bytes, dtype: str, shape: tuple[int, ...]) -> Any:
    import struct

    fmt_map = {"float32": "f", "float64": "d", "int32": "i", "int64": "q", "uint8": "B"}
    if dtype not in fmt_map:
        return list(raw)
    code = fmt_map[dtype]
    count = 1
    for s in shape:
        count *= s
    flat = list(struct.unpack(f"<{count}{code}", raw))
    if not shape:
        return flat[0]

    def reshape(seq: list[Any], dims: tuple[int, ...]) -> Any:
        if len(dims) == 1:
            return seq
        step = len(seq) // dims[0]
        return [reshape(seq[i * step : (i + 1) * step], dims[1:]) for i in range(dims[0])]

    return reshape(flat, shape)


class BasePolicy:
    """Unified interface implemented by all 4 method families (IL, RL, Agentic, RSI)."""

    policy_id: str = "base_policy"
    family: str = "agentic"
    control_mode: str = "osc_delta_eef"

    def reset(self, task_meta: dict[str, Any]) -> None:
        """Reset policy internal state at the start of an episode."""
        raise NotImplementedError

    def act(self, obs_dict: dict[str, Any]) -> tuple[list[float] | list[list[float]], dict[str, Any]]:
        """Return single-step or chunked action together with optional call_cost dict."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any background resources held by the policy."""
        return


_ABORT_SENTINEL = object()
_CMD_ACTION = "action"
_CMD_DONE = "done"


class _EpisodeAborted(BaseException):
    """Internal signal raised in imperative worker thread when an episode resets or closes."""


class RobotCodeAPI:
    """Closed-loop imperative robot API exposed to Code-as-Policies / RSI Python scripts."""

    def __init__(
        self,
        initial_obs: dict[str, Any],
        obs_queue: queue.Queue[Any],
        cmd_queue: queue.Queue[Any],
        abort_event: threading.Event,
        *,
        control_mode: str = "macro_target_eef",
    ) -> None:
        self._obs: dict[str, Any] = dict(initial_obs)
        self._obs_queue = obs_queue
        self._cmd_queue = cmd_queue
        self._abort_event = abort_event
        self._control_mode = normalize_control_mode(control_mode)
        self._grip: float = -1.0
        self._pending_cost: dict[str, int] = {
            "llm_unique_calls": 0,
            "llm_total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }

    def get_obs(self) -> dict[str, Any]:
        """Return the latest sanitized observation dictionary without advancing simulation."""
        if self._abort_event.is_set():
            raise _EpisodeAborted()
        return self._obs

    def record_llm_cost(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        unique_calls: int = 1,
        total_calls: int = 1,
    ) -> None:
        """Record LLM token/call usage to be attached to the next control command."""
        self._pending_cost["llm_unique_calls"] += int(unique_calls)
        self._pending_cost["llm_total_calls"] += int(total_calls)
        self._pending_cost["prompt_tokens"] += int(prompt_tokens)
        self._pending_cost["completion_tokens"] += int(completion_tokens)
        self._pending_cost["reasoning_tokens"] += int(reasoning_tokens)

    def _consume_pending_cost(self, *, macro_inc: int) -> dict[str, Any]:
        cost: dict[str, Any] = dict(self._pending_cost)
        cost["macro_turn_increment"] = int(macro_inc)
        for k in self._pending_cost:
            self._pending_cost[k] = 0
        return cost

    def move_to(
        self,
        x: float,
        y: float,
        z: float,
        *,
        rx: float | None = None,
        ry: float | None = None,
        rz: float | None = None,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        """Move end-effector to world coordinates (meters), optionally setting axis-angle ori & gripper."""
        if self._abort_event.is_set():
            raise _EpisodeAborted()
        if gripper is not None:
            self._grip = 1.0 if float(gripper) > 0.0 else -1.0

        has_ori = rx is not None and ry is not None and rz is not None
        if self._control_mode == "macro_rel_eef":
            cur_pos = self._obs.get("eef_pos", [0.0, 0.0, 0.5])
            rel_xyz = vec3_sub([float(x), float(y), float(z)], cur_pos)
            if has_ori:
                assert rx is not None and ry is not None and rz is not None
                cur_aa = quat2axisangle(self._obs.get("eef_quat", [0.0, 0.0, 0.0, 1.0]))
                rel_aa = [float(rx) - cur_aa[0], float(ry) - cur_aa[1], float(rz) - cur_aa[2]]
                action = rel_xyz + rel_aa + [self._grip]
            else:
                action = rel_xyz + [self._grip]
        else:
            if has_ori:
                assert rx is not None and ry is not None and rz is not None
                action = [float(x), float(y), float(z), float(rx), float(ry), float(rz), self._grip]
            else:
                action = [float(x), float(y), float(z), self._grip]

        self._cmd_queue.put((_CMD_ACTION, action, self._consume_pending_cost(macro_inc=1)))
        next_obs = self._obs_queue.get()
        if next_obs is _ABORT_SENTINEL or self._abort_event.is_set():
            raise _EpisodeAborted()
        self._obs = dict(next_obs)
        return self._obs

    def move_delta(
        self,
        dx: float,
        dy: float,
        dz: float,
        *,
        drx: float = 0.0,
        dry: float = 0.0,
        drz: float = 0.0,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        """Displace end-effector relative to its current world pose by metric (dx, dy, dz) and delta rot."""
        obs = self.get_obs()
        target_pos = vec3_add(obs["eef_pos"], [float(dx), float(dy), float(dz)])
        if abs(float(drx)) + abs(float(dry)) + abs(float(drz)) > 1e-8:
            target_quat = quat_multiply(axisangle2quat([drx, dry, drz]), obs["eef_quat"])
            aa = quat2axisangle(target_quat)
            return self.move_to(
                target_pos[0],
                target_pos[1],
                target_pos[2],
                rx=aa[0],
                ry=aa[1],
                rz=aa[2],
                gripper=gripper,
            )
        return self.move_to(target_pos[0], target_pos[1], target_pos[2], gripper=gripper)

    def open_gripper(self) -> dict[str, Any]:
        """Open gripper in place."""
        pos = self.get_obs()["eef_pos"]
        return self.move_to(pos[0], pos[1], pos[2], gripper=-1.0)

    def close_gripper(self) -> dict[str, Any]:
        """Close gripper in place."""
        pos = self.get_obs()["eef_pos"]
        return self.move_to(pos[0], pos[1], pos[2], gripper=1.0)


class ImperativePolicyBase(BasePolicy):
    """BasePolicy adapter enabling imperative Python Code-as-Policies scripts with if/while loops."""

    family: str = "agentic"
    control_mode: str = "macro_target_eef"

    def __init__(self) -> None:
        self._task_meta: dict[str, Any] = {}
        self._obs_queue: queue.Queue[Any] | None = None
        self._cmd_queue: queue.Queue[Any] | None = None
        self._abort_event: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._api: RobotCodeAPI | None = None
        self._started = False
        self._done = False
        self._waiting_obs = False
        self.last_episode_error: str | None = None

    def run_episode(self, api: RobotCodeAPI, task_meta: dict[str, Any]) -> None:
        """Override this method (or `exec` LLM-generated Python code against `api`)."""
        raise NotImplementedError

    def _stop_worker(self) -> None:
        if self._abort_event is not None:
            self._abort_event.set()
        if self._obs_queue is not None:
            try:
                self._obs_queue.put_nowait(_ABORT_SENTINEL)
            except queue.Full:
                pass
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._worker = None
        self._obs_queue = None
        self._cmd_queue = None
        self._abort_event = None
        self._api = None
        self._started = False
        self._done = False
        self._waiting_obs = False

    def close(self) -> None:
        self._stop_worker()

    def _hold_action(self, obs_dict: dict[str, Any]) -> list[float]:
        grip = self._api._grip if self._api is not None else -1.0
        if normalize_control_mode(self.control_mode) == "macro_rel_eef":
            return [0.0, 0.0, 0.0, grip]
        pos = obs_dict.get("eef_pos", [0.0, 0.0, 0.5])
        return [float(pos[0]), float(pos[1]), float(pos[2]), grip]

    def reset(self, task_meta: dict[str, Any]) -> None:
        self._stop_worker()
        self.last_episode_error = None
        self._task_meta = dict(task_meta)

    def act(self, obs_dict: dict[str, Any]) -> tuple[list[float] | list[list[float]], dict[str, Any]]:
        if self._done:
            return self._hold_action(obs_dict), {"macro_turn_increment": 0}

        if not self._started:
            obs_q: queue.Queue[Any] = queue.Queue(maxsize=1)
            cmd_q: queue.Queue[Any] = queue.Queue(maxsize=1)
            abort_ev = threading.Event()
            api = RobotCodeAPI(obs_dict, obs_q, cmd_q, abort_ev, control_mode=self.control_mode)
            self._obs_queue = obs_q
            self._cmd_queue = cmd_q
            self._abort_event = abort_ev
            self._api = api
            self._started = True
            self._waiting_obs = False
            task_meta = dict(self._task_meta)

            def _worker_fn() -> None:
                try:
                    self.run_episode(api, task_meta)
                except _EpisodeAborted:
                    pass
                except Exception as exc:
                    self.last_episode_error = f"{type(exc).__name__}: {exc}"
                finally:
                    cmd_q.put((_CMD_DONE, api._consume_pending_cost(macro_inc=0)))

            self._worker = threading.Thread(target=_worker_fn, daemon=True)
            self._worker.start()
        else:
            assert self._obs_queue is not None
            if self._waiting_obs:
                self._waiting_obs = False
                self._obs_queue.put(obs_dict)

        assert self._cmd_queue is not None
        kind, *payload = self._cmd_queue.get()
        if kind == _CMD_ACTION:
            action, cost = payload
            self._waiting_obs = True
            return action, cost

        leftover_cost = payload[0]
        self._done = True
        self._waiting_obs = False
        return self._hold_action(obs_dict), leftover_cost


class PolicyDispatcher:
    """Protocol router shared by ThreadingHTTPServer and direct in-memory testing."""

    def __init__(self, policy: BasePolicy) -> None:
        self.policy = policy
        self._lock = threading.Lock()

    def dispatch(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        if method == "GET" and path == "/health":
            return 200, {
                "status": "ok",
                "policy_id": self.policy.policy_id,
                "family": self.policy.family,
                "control_mode": normalize_control_mode(getattr(self.policy, "control_mode", "osc_delta_eef")),
            }
        if method == "POST" and path == "/reset":
            with self._lock:
                self.policy.reset(body or {})
            return 200, {"status": "reset_ok"}
        if method == "POST" and path == "/act":
            obs = _decode_value((body or {}).get("obs", {}))
            with self._lock:
                action, cost = self.policy.act(obs)
            return 200, {
                "action": _encode_value(action),
                "call_cost": cost or {},
            }
        return 404, {"error": f"Unknown endpoint {method} {path}"}


class PolicyHTTPServer:
    """Lightweight localhost HTTP server decoupling harness and GPU policy Conda envs."""

    def __init__(self, policy: BasePolicy, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.policy = policy
        self.dispatcher = PolicyDispatcher(policy)
        dispatcher = self.dispatcher

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                status, resp = dispatcher.dispatch("GET", self.path, None)
                self._send_json(status, resp)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                status, resp = dispatcher.dispatch("POST", self.path, payload)
                self._send_json(status, resp)

            def _send_json(self, status: int, resp: dict[str, Any]) -> None:
                data = json.dumps(resp).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return  # Keep harness stdout clean

        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self.host = self._httpd.server_address[0]
        self.port = int(self._httpd.server_address[1])
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start_background(self) -> None:
        """Run server loop in a daemon thread."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        """Block and serve policy requests until interrupted."""
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        """Stop server, notify policy close, and release socket."""
        self.policy.close()
        self._httpd.shutdown()
        self._httpd.server_close()


class RemotePolicyClient(BasePolicy):
    """Client adapter allowing harness/evaluate.py to query any external PolicyHTTPServer."""

    def __init__(self, base_url: str, timeout_sec: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        info = self._get("/health")
        self.policy_id = str(info.get("policy_id", "remote_policy"))
        self.family = str(info.get("family", "il"))
        self.control_mode = normalize_control_mode(info.get("control_mode", "osc_delta_eef"))

    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Policy server {path} failed ({exc.code}): {detail}") from exc

    def reset(self, task_meta: dict[str, Any]) -> None:
        self._post("/reset", task_meta)

    def act(self, obs_dict: dict[str, Any]) -> tuple[list[float] | list[list[float]], dict[str, Any]]:
        resp = self._post("/act", {"obs": _encode_value(obs_dict)})
        return resp["action"], resp.get("call_cost", {})
