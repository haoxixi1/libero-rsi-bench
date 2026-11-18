"""Pure-Python PNG keyframe and optional H.264 MP4 rollout recorder."""

from __future__ import annotations

import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def compose_dual_view_frame(obs: dict[str, Any], *, divider_px: int = 4) -> Any | None:
    """Compose side-by-side RGB frame [agentview | divider | wrist] from an observation dict."""
    imgs = obs.get("images") or {}
    agent = imgs.get("agentview")
    wrist = imgs.get("wrist")
    if agent is None and wrist is None:
        return None

    if np is not None:
        a = np.asarray(agent if agent is not None else wrist, dtype=np.uint8)
        w = np.asarray(wrist if wrist is not None else agent, dtype=np.uint8)
        if a.ndim != 3 or w.ndim != 3 or a.shape[0] != w.shape[0] or a.shape[2] != 3 or w.shape[2] != 3:
            return None
        div = np.full((a.shape[0], divider_px, 3), 32, dtype=np.uint8)
        return np.ascontiguousarray(np.concatenate([a, div, w], axis=1))

    a_list = agent if agent is not None else wrist
    w_list = wrist if wrist is not None else agent
    if not a_list or not w_list or len(a_list) != len(w_list):
        return None
    div_row = [[32, 32, 32] for _ in range(divider_px)]
    return [list(r_a) + div_row + list(r_w) for r_a, r_w in zip(a_list, w_list)]


def _frame_dimensions_and_rgb_bytes(frame: Any) -> tuple[int, int, bytes]:
    if np is not None and isinstance(frame, np.ndarray):
        arr = np.ascontiguousarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB frame, got shape {arr.shape}")
        return int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()

    height = len(frame)
    width = len(frame[0]) if height > 0 else 0
    buf = bytearray(height * width * 3)
    idx = 0
    for row in frame:
        for px in row:
            buf[idx] = max(0, min(255, int(px[0])))
            buf[idx + 1] = max(0, min(255, int(px[1])))
            buf[idx + 2] = max(0, min(255, int(px[2])))
            idx += 3
    return height, width, bytes(buf)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_rgb_png(path: Path, frame: Any) -> Path:
    """Write an HxWx3 uint8 RGB frame to a PNG file using Python stdlib only."""
    height, width, rgb_bytes = _frame_dimensions_and_rgb_bytes(frame)
    stride = width * 3
    scanlines = bytearray(height * (stride + 1))
    for y in range(height):
        src_off = y * stride
        dst_off = y * (stride + 1)
        scanlines[dst_off] = 0
        scanlines[dst_off + 1 : dst_off + 1 + stride] = rgb_bytes[src_off : src_off + stride]

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(scanlines), level=6)
    png_data = _PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_data)
    return out_path


def save_episode_keyframes(
    keyframes_dir: Path,
    episode_id: str,
    frames: list[Any],
    *,
    suffix: str = "",
) -> list[Path]:
    """Save start / middle / end RGB PNG keyframes for an episode rollout."""
    if not frames:
        return []
    out_dir = Path(keyframes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [0, len(frames) // 2, len(frames) - 1]
    labels = ("00_start", "01_mid", "02_end")
    saved: list[Path] = []
    tag = f"_{suffix}" if suffix else ""
    for label, idx in zip(labels, indices):
        fname = f"{episode_id}_{label}{tag}.png"
        saved.append(write_rgb_png(out_dir / fname, frames[idx]))
    return saved


def write_mp4(path: Path, frames: list[Any], *, fps: int = 20) -> Path | None:
    """Encode a sequence of HxWx3 frames into an H.264 yuv420p MP4 video if supported."""
    if not frames:
        return None
    height, width, _ = _frame_dimensions_and_rgb_bytes(frames[0])
    if height < 8 or width < 8 or height % 2 != 0 or width % 2 != 0:
        return None

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "21",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for fr in frames:
            h_i, w_i, raw = _frame_dimensions_and_rgb_bytes(fr)
            if (h_i, w_i) != (height, width):
                continue
            proc.stdin.write(raw)
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}: {stderr}")
        return out_path

    try:  # pragma: no cover
        import imageio.v2 as imageio  # type: ignore[import-not-found]

        imageio.mimsave(str(out_path), frames, fps=int(fps))
        return out_path
    except Exception:
        return None
