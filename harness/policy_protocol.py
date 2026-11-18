"""Decoupled cross-environment Policy protocol (BasePolicy + JSON HTTP Server/Client)."""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

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
        """Stop server and release socket."""
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
