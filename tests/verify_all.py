"""Comprehensive end-to-end pure-Python verification suite for libero-rsi-bench."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.minimal_code_policy import MinimalCodeAsPolicy
from examples.minimal_policy_server import MinimalTemplatePolicy
from harness.budget_tracker import (
    BudgetExceededError,
    OnlineBudgetGuard,
    SealedAccessViolationError,
)
from harness.dev_env import DevInteractionEnv
from harness.env_adapter import create_env_adapter, rollout_episode
from harness.evaluate import ScriptedSmokePolicy, run_evaluation
from harness.math_kinematics import (
    axisangle2quat,
    deproject_pixel_to_world,
    deterministic_gaussian_vec3,
    quat2axisangle,
    quat_multiply,
    quat_orientation_error_aa,
)
from harness.policy_protocol import (
    BasePolicy,
    ImperativePolicyBase,
    PolicyDispatcher,
    PolicyHTTPServer,
    RobotCodeAPI,
)
from harness.stats_and_report import (
    exact_mcnemar_test,
    generate_comparison_report,
    mcnemar_power,
    wilson_ci_95,
)
from harness.video_recorder import compose_dual_view_frame, write_rgb_png
from protocol.generate_splits import DEFAULT_LOCK_PATH, verify_lock


def test_splits_lock() -> None:
    print("[1/9] Verifying splits.lock.json & SHA-256 protocol...")
    proc = subprocess.run(
        [sys.executable, str(ROOT_DIR / "protocol" / "generate_splits.py"), "--verify"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "VERIFIED" in proc.stdout
    lock_data = json.loads(DEFAULT_LOCK_PATH.read_text(encoding="utf-8"))
    verify_lock(lock_data)
    assert lock_data["counts"] == {"burned": 8, "dev": 100, "sealed_id": 100, "sealed_ood": 112}
    # Confirm clean_base_name is present on every split item
    for name in ("burned", "dev", "sealed_id", "sealed_ood"):
        assert all(ep.get("clean_base_name") for ep in lock_data["splits"][name])
    print("  -> OK:", proc.stdout.strip())


def test_math_kinematics() -> None:
    print("[2/9] Verifying quaternion/axis-angle round-trip & SE(3) deprojection...")
    test_vectors = [
        [0.0, 0.0, 0.0],
        [1e-9, -2e-9, 1e-9],
        [0.4, -0.7, 0.9],
        [1.5, 0.3, -0.8],
        [-1.8, 1.1, 0.4],
    ]
    for v in test_vectors:
        q = axisangle2quat(v)
        v_rec = quat2axisangle(q)
        q_rec = axisangle2quat(v_rec)
        dot = abs(sum(a * b for a, b in zip(q, q_rec)))
        assert abs(1.0 - dot) < 1e-6, f"Round-trip quat error for {v}: dot={dot}"
        if math.sqrt(sum(x * x for x in v)) <= math.pi:
            err = max(abs(a - b) for a, b in zip(v, v_rec))
            assert err < 1e-6, f"Axis-angle mismatch for {v}: got {v_rec}, err={err}"

    K = [[500.0, 0.0, 128.0], [0.0, 500.0, 128.0], [0.0, 0.0, 1.0]]
    cam2world = [
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
    world_pt = deproject_pixel_to_world(228.0, 128.0, 2.0, K, cam2world)
    expected = [1.0, 2.4, 2.5]
    max_err = max(abs(a - b) for a, b in zip(world_pt, expected))
    assert max_err < 1e-6, f"Deprojection mismatch: {world_pt} vs {expected}"

    n1 = deterministic_gaussian_vec3(2025, 17, 0.01)
    n2 = deterministic_gaussian_vec3(2025, 17, 0.01)
    assert n1 == n2
    print("  -> OK: kinematics & SE(3) errors < 1e-6")


def test_budget_and_sealed_protection() -> None:
    print("[3/9] Verifying OnlineBudgetGuard, DevInteractionEnv & sealed-split protection...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sealed_log = tmp_dir / "sealed_reads.log"

        guard = OnlineBudgetGuard(max_online_episodes=3, sealed_log_path=sealed_log)
        guard.consume_online_episode(steps=100)
        guard.consume_online_episode(steps=100)
        guard.consume_online_episode(steps=100)
        try:
            guard.consume_online_episode(steps=10)
            raise AssertionError("Expected BudgetExceededError")
        except BudgetExceededError:
            pass

        dev_env = DevInteractionEnv(
            task_key="spatial_t0",
            env_backend="mock",
            max_online_episodes=2,
            allow_dev_privileged_info=True,
        )
        obs, info = dev_env.reset()
        assert info["remaining_budget"] == 1 and "instruction" in obs
        assert "dev_privileged" in info and "dev_privileged" not in obs
        assert "task_type" not in obs and "holding_object" not in obs
        _, _, term, trunc, s_info = dev_env.step([0.0] * 6 + [-1.0])
        assert not term and not trunc and "dev_privileged" in s_info
        dev_env.reset()
        try:
            dev_env.reset()
            raise AssertionError("Expected BudgetExceededError from DevInteractionEnv")
        except BudgetExceededError:
            pass
        finally:
            dev_env.close()

        # Verify hard max_macro_turns enforcement for agentic policies
        class GreedyAgenticPolicy(BasePolicy):
            policy_id = "greedy_agentic"
            family = "agentic"

            def reset(self, task_meta):
                assert "task_type" not in task_meta

            def act(self, obs_dict):
                assert "task_type" not in obs_dict and "holding_object" not in obs_dict
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], {"macro_turn_increment": 1}

        mock_env = create_env_adapter("mock", perception_mode="rgbd_projection")
        try:
            ep_res = rollout_episode(
                mock_env,
                GreedyAgenticPolicy(),
                {"episode_id": "t_cap", "task_key": "spatial_t0", "split": "burned", "seed": 1},
                {"instruction": "test", "task_type": "pick_place", "standard_steps": 220, "extended_steps": 40, "max_macro_turns": 5, "hold_success_steps": 1},
            )
            assert ep_res.macro_turns == 5, f"Expected cap 5, got {ep_res.macro_turns}"
        finally:
            mock_env.close()

        # Accessing sealed_id without --confirm-unseal must raise SealedAccessViolationError
        try:
            run_evaluation(
                split="sealed_id",
                out_dir=tmp_dir / "unauthorized",
                env_backend="mock",
                dummy_policy="zero",
                confirm_unseal=False,
                sealed_log_path=sealed_log,
                limit=4,
            )
            raise AssertionError("Expected SealedAccessViolationError")
        except SealedAccessViolationError:
            pass

        # Accessing sealed_id WITH --confirm-unseal succeeds and logs SHA-256 audit entry
        summary = run_evaluation(
            split="sealed_id",
            out_dir=tmp_dir / "authorized_sealed",
            env_backend="mock",
            dummy_policy="scripted_smoke",
            perception_mode="noisy_keypoints",
            confirm_unseal=True,
            sealed_log_path=sealed_log,
            limit=8,
        )
        assert summary["num_episodes"] == 8
        log_text = sealed_log.read_text(encoding="utf-8")
        assert "split=sealed_id" in log_text and summary["sealed_sha256"] in log_text
    print("  -> OK: budget cap, DevInteractionEnv & sealed-split audit verified")


def test_http_server_and_chunked_rollouts() -> None:
    print("[4/9] Verifying PolicyHTTPServer <-> RemotePolicyClient (single-step & chunked)...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        chunk_policy = MinimalTemplatePolicy(family="il", chunk_size=10)
        smoke_policy = ScriptedSmokePolicy()
        http_used = False
        try:
            srv_chunk = PolicyHTTPServer(chunk_policy, host="127.0.0.1", port=0)
            srv_chunk.start_background()
            srv_smoke = PolicyHTTPServer(smoke_policy, host="127.0.0.1", port=0)
            srv_smoke.start_background()
            http_used = True
            try:
                summary_chunk = run_evaluation(
                    split="burned",
                    out_dir=tmp_dir / "chunk_http_burned",
                    env_backend="mock",
                    perception_mode="rgbd_projection",
                    policy_url=srv_chunk.url,
                )
                assert summary_chunk["num_episodes"] == 8

                summary_smoke_http = run_evaluation(
                    split="dev",
                    out_dir=tmp_dir / "smoke_http_dev",
                    env_backend="mock",
                    perception_mode="noisy_keypoints",
                    policy_url=srv_smoke.url,
                    limit=12,
                )
                assert summary_smoke_http["num_episodes"] == 12
            finally:
                srv_chunk.shutdown()
                srv_smoke.shutdown()
        except PermissionError:
            disp = PolicyDispatcher(chunk_policy)
            st, payload = disp.dispatch("GET", "/health")
            assert st == 200 and payload["family"] == "il"

        # Also run full 100-episode Dev comparison across built-in harness test policies & --policy-class
        s_zero = run_evaluation(
            split="dev",
            out_dir=tmp_dir / "zero_dev",
            env_backend="mock",
            dummy_policy="zero",
        )
        s_rand = run_evaluation(
            split="dev",
            out_dir=tmp_dir / "random_dev",
            env_backend="mock",
            dummy_policy="random",
        )
        s_smoke = run_evaluation(
            split="dev",
            out_dir=tmp_dir / "smoke_dev",
            env_backend="mock",
            perception_mode="noisy_keypoints",
            dummy_policy="scripted_smoke",
        )
        s_cls = run_evaluation(
            split="burned",
            out_dir=tmp_dir / "class_burned",
            env_backend="mock",
            policy_class="examples.minimal_policy_server:MinimalTemplatePolicy",
            policy_kwargs='{"family": "il", "chunk_size": 5}',
        )
        assert s_cls["num_episodes"] == 8

        for name, s in [("zero", s_zero), ("random", s_rand), ("scripted_smoke", s_smoke)]:
            assert s["num_episodes"] == 100
            assert s["sr_1_0x"] <= s["sr_1_5x"] + 1e-9, f"{name}: SR@1.0x > SR@1.5x"

        eps = [
            json.loads(line)
            for line in (tmp_dir / "smoke_dev" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(eps) == 100
        assert sum(Counter(e["failure_cluster"] for e in eps).values()) == 100
        assert s_smoke["sr_1_5x"] > s_zero["sr_1_5x"]

        # Verify report & SVG generation across runs
        report_dir = tmp_dir / "report"
        report_md = generate_comparison_report(
            [tmp_dir / "zero_dev", tmp_dir / "random_dev", tmp_dir / "smoke_dev"],
            report_dir,
        )
        text = report_md.read_text(encoding="utf-8")
        assert "Main Results & Lifecycle Cost" in text and "Paired Exact McNemar Tests" in text
        for svg_name in ("horizon_curve.svg", "failure_clusters.svg", "cost_pareto.svg"):
            svg = (report_dir / svg_name).read_text(encoding="utf-8")
            assert svg.startswith("<svg") and svg.strip().endswith("</svg>")

        print(
            f"  -> OK ({'HTTP + Chunked' if http_used else 'Dispatcher fallback'}): "
            f"zero={s_zero['sr_1_5x']*100:.1f}%, rand={s_rand['sr_1_5x']*100:.1f}%, "
            f"scripted_smoke={s_smoke['sr_1_5x']*100:.1f}% (1.0x={s_smoke['sr_1_0x']*100:.1f}%)"
        )


def test_examples_cli() -> None:
    print("[5/9] Verifying examples/minimal_dev_interaction.py CLI...")
    proc = subprocess.run(
        [sys.executable, str(ROOT_DIR / "examples" / "minimal_dev_interaction.py"), "--env-backend", "mock"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Budget guard verified" in proc.stdout
    print("  -> OK: minimal_dev_interaction.py passed")


def test_stats_functions() -> None:
    print("[6/9] Verifying Wilson CI, Exact McNemar, and Finite-Sample Power...")
    p, lo, hi = wilson_ci_95(80, 100)
    assert abs(p - 0.80) < 1e-9 and 0.70 < lo < 0.75 and 0.85 < hi < 0.90
    p_eq = exact_mcnemar_test(11, 11, alternative="two-sided")
    assert abs(p_eq - 1.0) < 1e-9
    p_sig = exact_mcnemar_test(18, 2, alternative="two-sided")
    assert p_sig < 0.001
    pw_1 = mcnemar_power(100, 1.0, discordant_base=0.06, alpha=0.05, alternative="two-sided")
    pw_15 = mcnemar_power(100, 15.0, discordant_base=0.06, alpha=0.05, alternative="two-sided")
    assert pw_1 < 0.08 and pw_15 > 0.70
    print(f"  -> OK: Power(+1pt)={pw_1*100:.1f}%, Power(+15pt)={pw_15*100:.1f}%")


def test_multi_control_modes_and_joints() -> None:
    print("[7/9] Verifying joint proprioception & all control modes (abs_eef, joint, macro_rel/target)...")
    # 1. Quaternion algebra checks
    q_cur = axisangle2quat([0.2, -0.3, 0.5])
    dq_world = axisangle2quat([0.1, 0.25, -0.15])
    q_tgt = quat_multiply(dq_world, q_cur)
    aa_err = quat_orientation_error_aa(q_tgt, q_cur)
    assert max(abs(a - b) for a, b in zip(aa_err, [0.1, 0.25, -0.15])) < 1e-6

    ep = {"episode_id": "ctrl_t0", "task_key": "object_t0", "split": "burned", "seed": 1001}
    cfg = {
        "instruction": "pick up the alphabet soup and place it in the basket",
        "standard_steps": 280,
        "extended_steps": 450,
        "max_macro_turns": 15,
        "hold_success_steps": 1,
    }

    # 2. Step-level abs_eef (7D)
    env_abs = create_env_adapter("mock", perception_mode="noisy_keypoints", control_mode="abs_eef")
    try:
        obs = env_abs.reset_episode(ep, cfg)
        assert len(obs["joint_pos"]) == 7 and len(obs["joint_vel"]) == 7 and len(obs["gripper_qpos"]) == 2
        tgt_pos = [obs["eef_pos"][0] + 0.06, obs["eef_pos"][1] - 0.04, obs["eef_pos"][2] - 0.05]
        tgt_aa = quat2axisangle(obs["eef_quat"])
        for _ in range(10):
            obs, _, _, _ = env_abs.step(tgt_pos + tgt_aa + [-1.0])
        assert max(abs(obs["eef_pos"][i] - tgt_pos[i]) for i in range(3)) < 1e-5
    finally:
        env_abs.close()

    # 3. DevInteractionEnv + 8D abs_joint_pos & delta_joint_pos (must NOT truncate 8th dim)
    dev_j = DevInteractionEnv(task_key="object_t0", env_backend="mock", control_mode="abs_joint_pos", max_online_episodes=2)
    try:
        j_obs, _ = dev_j.reset()
        assert len(j_obs["joint_pos"]) == 7 and len(j_obs["joint_vel"]) == 7
        q_target = [j_obs["joint_pos"][i] + (0.08 if i % 2 == 0 else -0.06) for i in range(7)]
        for _ in range(5):
            j_obs, _, _, _, _ = dev_j.step(q_target + [1.0])
        assert max(abs(j_obs["joint_pos"][i] - q_target[i]) for i in range(7)) < 1e-6
    finally:
        dev_j.close()

    env_dj = create_env_adapter("mock", perception_mode="noisy_keypoints", control_mode="delta_joint_pos")
    try:
        d_obs = env_dj.reset_episode(ep, cfg)
        q0 = list(d_obs["joint_pos"])
        d_obs, _, _, _ = env_dj.step([0.5, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        assert abs(d_obs["joint_pos"][0] - (q0[0] + 0.025)) < 1e-6
        assert abs(d_obs["joint_pos"][1] - (q0[1] - 0.025)) < 1e-6
    finally:
        env_dj.close()

    # 4. Macro closed-loop policies (BOTH macro_target_eef and macro_rel_eef) via HTTP /health negotiation
    class MacroPickPlacePolicy(BasePolicy):
        def __init__(self, mode: str) -> None:
            self.control_mode = mode
            self.policy_id = f"macro_{mode}_smoke"
            self.family = "agentic"
            self._stage = 0

        def reset(self, task_meta: dict) -> None:
            self._stage = 0

        def act(self, obs_dict: dict) -> tuple[list[float], dict]:
            kp = obs_dict["keypoints"]
            tgt = kp["target_xyz"]
            goal = kp["goal_xyz"]
            eef = obs_dict["eef_pos"]
            waypoints = [
                (tgt[0], tgt[1], tgt[2] + 0.14, -1.0),
                (tgt[0], tgt[1], tgt[2], -1.0),
                (tgt[0], tgt[1], tgt[2], 1.0),
                (tgt[0], tgt[1], goal[2] + 0.14, 1.0),
                (goal[0], goal[1], goal[2] + 0.12, 1.0),
                (goal[0], goal[1], goal[2] + 0.10, -1.0),
            ]
            idx = min(self._stage, len(waypoints) - 1)
            self._stage += 1
            wx, wy, wz, grip = waypoints[idx]
            if self.control_mode == "macro_rel_eef":
                act = [wx - eef[0], wy - eef[1], wz - eef[2], grip]
            else:
                act = [wx, wy, wz, grip]
            return act, {"macro_turn_increment": 1}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for mode in ("macro_target_eef", "macro_rel_eef"):
            pol = MacroPickPlacePolicy(mode)
            try:
                srv = PolicyHTTPServer(pol, host="127.0.0.1", port=0)
                srv.start_background()
                try:
                    summary = run_evaluation(
                        split="dev",
                        out_dir=tmp_dir / f"http_{mode}",
                        env_backend="mock",
                        perception_mode="noisy_keypoints",
                        noise_sigma_mm=3.0,
                        policy_url=srv.url,
                        task_filter=["object_t0", "spatial_t0"],
                        limit=10,
                    )
                finally:
                    srv.shutdown()
            except PermissionError:
                disp = PolicyDispatcher(pol)
                st, payload = disp.dispatch("GET", "/health")
                assert st == 200 and payload["control_mode"] == mode
                env_direct = create_env_adapter("mock", perception_mode="noisy_keypoints", noise_sigma_mm=3.0, control_mode=mode)
                try:
                    direct_results = [
                        rollout_episode(env_direct, pol, {**ep, "task_key": "object_t0", "seed": 2100 + i}, cfg)
                        for i in range(6)
                    ]
                finally:
                    env_direct.close()
                summary = {
                    "control_mode": mode,
                    "sr_1_5x": sum(1 for r in direct_results if r.success_1_5x) / len(direct_results),
                    "mean_macro_turns": sum(r.macro_turns for r in direct_results) / len(direct_results),
                }
            assert summary["control_mode"] == mode
            assert summary["sr_1_5x"] >= 0.8, f"Expected high SR for {mode}, got {summary['sr_1_5x']}"
            assert 4.0 <= summary["mean_macro_turns"] <= 8.0

    print("  -> OK: joint_pos/vel + abs_eef + abs/delta_joint + macro_target_eef & macro_rel_eef verified")


def test_video_and_keyframe_recorder() -> None:
    print("[8/9] Verifying pure-Python PNG keyframe writer & evaluate --save-keyframes integration...")
    img_a = [[[20, 120, 220] for _ in range(8)] for _ in range(8)]
    img_w = [[[220, 80, 40] for _ in range(8)] for _ in range(8)]
    dual = compose_dual_view_frame({"images": {"agentview": img_a, "wrist": img_w}})
    assert dual is not None and len(dual) == 8 and len(dual[0]) == 20

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        png_path = write_rgb_png(tmp_dir / "test_dual.png", dual)
        raw = png_path.read_bytes()
        assert raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) > 32

        summary = run_evaluation(
            split="burned",
            out_dir=tmp_dir / "media_run",
            env_backend="mock",
            perception_mode="rgbd_projection",
            dummy_policy="zero",
            limit=2,
            save_keyframes=True,
        )
        assert summary["num_episodes"] == 2
        pngs = sorted((tmp_dir / "media_run" / "keyframes").glob("*.png"))
        assert len(pngs) == 6, f"Expected 6 keyframe PNGs (3 per episode), got {len(pngs)}"
        assert all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in pngs)
    print("  -> OK: PNG encoder & rollout keyframe export verified")


def test_code_as_policies_bridge() -> None:
    print("[9/9] Verifying Code-as-Policies ImperativePolicyBase (HTTP, closed-loop exec, error safety)...")
    import threading

    class ExecCapPolicy(ImperativePolicyBase):
        policy_id = "exec_cap_test"
        family = "agentic"
        control_mode = "macro_target_eef"

        def run_episode(self, api: RobotCodeAPI, task_meta: dict) -> None:
            api.record_llm_cost(prompt_tokens=250, completion_tokens=80, unique_calls=1, total_calls=1)
            code = """
obs = api.get_obs()
tgt = obs["keypoints"]["target_xyz"]
goal = obs["keypoints"]["goal_xyz"]
obs = api.move_to(tgt[0], tgt[1], tgt[2] + 0.14)
obs = api.move_to(tgt[0], tgt[1], tgt[2])
obs = api.close_gripper()
if obs["gripper_opening"] < 0.02:
    obs = api.move_delta(0.0, 0.0, goal[2] + 0.14 - obs["eef_pos"][2])
obs = api.move_to(goal[0], goal[1], goal[2] + 0.12)
obs = api.open_gripper()
"""
            exec(code, {"api": api, "task_meta": task_meta})

    class FaultyCapPolicy(ImperativePolicyBase):
        policy_id = "faulty_cap_test"
        family = "agentic"
        control_mode = "macro_target_eef"

        def run_episode(self, api: RobotCodeAPI, task_meta: dict) -> None:
            api.record_llm_cost(prompt_tokens=120, completion_tokens=30, unique_calls=1, total_calls=1)
            _ = 1 / 0  # Simulated bug inside LLM-generated script

    base_threads = threading.active_count()
    pol = ExecCapPolicy()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            try:
                srv = PolicyHTTPServer(pol, host="127.0.0.1", port=0)
                srv.start_background()
                try:
                    summary = run_evaluation(
                        split="dev",
                        out_dir=tmp_dir / "cap_http",
                        env_backend="mock",
                        perception_mode="noisy_keypoints",
                        noise_sigma_mm=2.0,
                        policy_url=srv.url,
                        task_filter=["object_t0", "spatial_t0"],
                        limit=6,
                    )
                finally:
                    srv.shutdown()
            except PermissionError:
                env_m = create_env_adapter("mock", perception_mode="noisy_keypoints", noise_sigma_mm=2.0, control_mode="macro_target_eef")
                try:
                    ep0 = {"episode_id": "cap_0", "task_key": "object_t0", "split": "burned", "seed": 1001}
                    cfg0 = {"instruction": "test", "standard_steps": 280, "extended_steps": 450, "max_macro_turns": 15, "hold_success_steps": 1}
                    res = rollout_episode(env_m, pol, ep0, cfg0)
                    assert res.success_1_5x and res.llm_calls > 0
                finally:
                    env_m.close()
                summary = {"sr_1_5x": 1.0, "cost_ledger": {"prompt_tokens": res.tokens}}
            assert summary["sr_1_5x"] >= 0.8, f"Expected CaP SR >= 0.8, got {summary}"
            assert summary["cost_ledger"]["prompt_tokens"] >= 250
            assert pol.last_episode_error is None
    finally:
        pol.close()

    # Check MinimalCodeAsPolicy instantiation & error containment on FaultyCapPolicy
    sample_cap = MinimalCodeAsPolicy()
    faulty = FaultyCapPolicy()
    env_f = create_env_adapter("mock", perception_mode="noisy_keypoints", control_mode="macro_target_eef")
    try:
        cfg_f = {"instruction": "test", "standard_steps": 220, "extended_steps": 80, "max_macro_turns": 5, "hold_success_steps": 1}
        ep_f = {"episode_id": "cap_err", "task_key": "object_t0", "split": "burned", "seed": 1002}
        res_f = rollout_episode(env_f, faulty, ep_f, cfg_f)
        assert faulty.last_episode_error and "ZeroDivisionError" in faulty.last_episode_error
        assert res_f.tokens == 150
        res_s = rollout_episode(env_f, sample_cap, ep_f, cfg_f)
        assert res_s.steps_taken > 0 and sample_cap.last_episode_error is None
    finally:
        sample_cap.close()
        faulty.close()
        env_f.close()

    assert threading.active_count() <= base_threads + 1
    print("  -> OK: Code-as-Policies HTTP + ImperativePolicyBase + error containment verified")


def main() -> None:
    test_splits_lock()
    test_math_kinematics()
    test_budget_and_sealed_protection()
    test_http_server_and_chunked_rollouts()
    test_examples_cli()
    test_stats_functions()
    test_multi_control_modes_and_joints()
    test_video_and_keyframe_recorder()
    test_code_as_policies_bridge()
    print("\nALL LOCAL VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
