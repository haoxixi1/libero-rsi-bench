"""Kinematics, SE(3) deprojection, and deterministic noise utilities.

Pure-Python implementations so the harness works identically in minimal Python
environments and on GPU servers (with optional NumPy helpers when needed).
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def vec3_norm(v: Sequence[float]) -> float:
    """Return Euclidean L2 norm of a 3D vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Return a - b for 3D vectors."""
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])]


def vec3_add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Return a + b for 3D vectors."""
    return [float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2])]


def vec3_dist(a: Sequence[float], b: Sequence[float]) -> float:
    """Return Euclidean distance between two 3D points."""
    return vec3_norm(vec3_sub(a, b))


def vec2_dist(a: Sequence[float], b: Sequence[float]) -> float:
    """Return horizontal XY Euclidean distance between two points."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def clip(x: float, lo: float, hi: float) -> float:
    """Clamp scalar x to [lo, hi]."""
    return lo if x < lo else (hi if x > hi else x)


def quat2axisangle(quat_xyzw: Sequence[float]) -> list[float]:
    """Convert xyzw quaternion to shortest-arc 3D axis-angle vector (radians)."""
    qx, qy, qz, qw = (float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3]))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Canonicalize to w >= 0 so rotation angle lies in [0, pi].
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    qw = clip(qw, -1.0, 1.0)
    sin_half = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_half < 1e-8:
        return [2.0 * qx, 2.0 * qy, 2.0 * qz]
    angle = 2.0 * math.acos(qw)
    scale = angle / sin_half
    return [qx * scale, qy * scale, qz * scale]


def axisangle2quat(axisangle: Sequence[float]) -> list[float]:
    """Convert 3D axis-angle vector (radians) to normalized xyzw quaternion."""
    ax, ay, az = float(axisangle[0]), float(axisangle[1]), float(axisangle[2])
    angle = math.sqrt(ax * ax + ay * ay + az * az)
    if angle < 1e-8:
        # First-order small-angle approximation sin(theta/2)/theta ~ 0.5
        qx, qy, qz = 0.5 * ax, 0.5 * ay, 0.5 * az
        qw = math.sqrt(max(0.0, 1.0 - (qx * qx + qy * qy + qz * qz)))
        return [qx, qy, qz, qw]
    half = 0.5 * angle
    s = math.sin(half) / angle
    return [ax * s, ay * s, az * s, math.cos(half)]


def quat_conjugate(quat_xyzw: Sequence[float]) -> list[float]:
    """Return the normalized conjugate [-qx, -qy, -qz, qw] of an xyzw quaternion."""
    qx, qy, qz, qw = (float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3]))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [-qx / norm, -qy / norm, -qz / norm, qw / norm]


def quat_multiply(q1_xyzw: Sequence[float], q2_xyzw: Sequence[float]) -> list[float]:
    """Compute normalized Hamilton product q1 * q2 in xyzw convention."""
    x1, y1, z1, w1 = (float(q1_xyzw[0]), float(q1_xyzw[1]), float(q1_xyzw[2]), float(q1_xyzw[3]))
    x2, y2, z2, w2 = (float(q2_xyzw[0]), float(q2_xyzw[1]), float(q2_xyzw[2]), float(q2_xyzw[3]))
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / norm, y / norm, z / norm, w / norm]


def quat_orientation_error_aa(
    target_quat_xyzw: Sequence[float],
    current_quat_xyzw: Sequence[float],
) -> list[float]:
    """Return shortest-arc world-frame axis-angle error from current to target quaternion."""
    q_err = quat_multiply(target_quat_xyzw, quat_conjugate(current_quat_xyzw))
    return quat2axisangle(q_err)


def deproject_pixel_to_world(
    u: float,
    v: float,
    depth: float,
    intrinsic_K: Sequence[Sequence[float]],
    cam2world_4x4: Sequence[Sequence[float]],
) -> list[float]:
    """Deproject pixel (u, v) and metric depth (m) into 3D world coordinates via SE(3).

    Assumes pinhole camera frame (+Z forward, +X right, +Y down) aligned with `cam2world_4x4`.
    """
    fx = float(intrinsic_K[0][0])
    cx = float(intrinsic_K[0][2])
    fy = float(intrinsic_K[1][1])
    cy = float(intrinsic_K[1][2])
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError(f"Invalid focal lengths in intrinsic_K: fx={fx}, fy={fy}")

    xc = (float(u) - cx) * float(depth) / fx
    yc = (float(v) - cy) * float(depth) / fy
    zc = float(depth)

    xw = (
        float(cam2world_4x4[0][0]) * xc
        + float(cam2world_4x4[0][1]) * yc
        + float(cam2world_4x4[0][2]) * zc
        + float(cam2world_4x4[0][3])
    )
    yw = (
        float(cam2world_4x4[1][0]) * xc
        + float(cam2world_4x4[1][1]) * yc
        + float(cam2world_4x4[1][2]) * zc
        + float(cam2world_4x4[1][3])
    )
    zw = (
        float(cam2world_4x4[2][0]) * xc
        + float(cam2world_4x4[2][1]) * yc
        + float(cam2world_4x4[2][2]) * zc
        + float(cam2world_4x4[2][3])
    )
    return [xw, yw, zw]


def deterministic_gaussian_vec3(seed: int, salt: int, sigma_m: float) -> list[float]:
    """Generate deterministic 3D zero-mean Gaussian noise vector from (seed, salt)."""
    if sigma_m <= 0.0:
        return [0.0, 0.0, 0.0]
    rng = random.Random(((int(seed) & 0xFFFFFFFF) << 32) ^ (int(salt) & 0xFFFFFFFF))
    return [rng.gauss(0.0, sigma_m), rng.gauss(0.0, sigma_m), rng.gauss(0.0, sigma_m)]
