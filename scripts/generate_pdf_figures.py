#!/usr/bin/env python3
"""Generate publication-grade vector PDF figures for docs/IL_EXPERIMENT_REPORT.tex."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from harness.stats_and_report import CLUSTERS_ORDER, wilson_ci_95

FIG_DIR = ROOT_DIR / "docs" / "figures"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_summary(run_name: str) -> dict[str, Any]:
    return json.loads((ROOT_DIR / "runs" / run_name / "summary.json").read_text(encoding="utf-8"))


def make_fig1_training_curves() -> None:
    log50 = _load_jsonl(ROOT_DIR / "checkpoints" / "il_act_50" / "train_log.jsonl")
    log20 = _load_jsonl(ROOT_DIR / "checkpoints" / "il_act_20" / "train_log.jsonl")

    ep50 = [r["epoch"] for r in log50]
    tr50 = [r["train_loss"] for r in log50]
    va50 = [r["val_loss"] for r in log50]
    lr50 = [r["lr"] * 1e4 for r in log50]

    ep20 = [r["epoch"] for r in log20]
    tr20 = [r["train_loss"] for r in log20]
    va20 = [r["val_loss"] for r in log20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax1.plot(ep50, tr50, color="#1f77b4", lw=2.0, label="il_act_50 Train L1 (165 demos -> 0.0518)")
    ax1.plot(ep20, tr20, color="#d62728", lw=2.0, linestyle="--", label="il_act_20 Train L1 (80 demos -> 0.0535)")
    ax1.set_title("Training L1 Action-Chunk Loss Convergence")
    ax1.set_xlabel("Epoch (Cosine LR Decay, 100 Epochs)")
    ax1.set_ylabel("Smooth L1 Loss")
    ax1.set_xlim(1, 100)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(frameon=False, loc="upper right")

    ax2.plot(ep50, va50, color="#1f77b4", lw=1.8, alpha=0.85, label="il_act_50 Held-out Val Loss (min=0.5058 @ Ep 4)")
    ax2.plot(ep20, va20, color="#d62728", lw=1.8, linestyle="--", alpha=0.85, label="il_act_20 Held-out Val Loss (min=0.4381 @ Ep 4)")
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.plot(ep50, lr50, color="#2ca02c", lw=1.3, linestyle=":", label="Cosine LR (x1e-4)")
    ax2_r.set_ylabel("Learning Rate (x1e-4)", color="#2ca02c")
    ax2_r.tick_params(axis="y", colors="#2ca02c")

    ax2.set_title("Open-Loop Val Loss vs. Cosine Learning Rate")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Held-out Open-Loop L1 Loss")
    ax2.set_xlim(1, 100)
    ax2.grid(True, linestyle=":", alpha=0.5)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=True, framealpha=0.9, edgecolor="#dddddd", loc="lower right", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig2_horizon_and_tasks() -> None:
    runs = [
        ("il_act_50_id", "il_act_50 TEST_ID (91.0%)", "#1f77b4", "-"),
        ("il_act_20_id", "il_act_20 TEST_ID (87.0%)", "#1f77b4", "--"),
        ("il_act_50_ood", "il_act_50 TEST_OOD (79.5%)", "#d62728", "-"),
        ("il_act_20_ood", "il_act_20 TEST_OOD (68.8%)", "#d62728", "--"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0))

    budgets = np.linspace(0, 750, 76)
    for rname, label, color, ls in runs:
        eps = _load_jsonl(ROOT_DIR / "runs" / rname / "episodes.jsonl")
        n = len(eps)
        ys = [
            100.0
            * sum(
                1
                for ep in eps
                if ep["success_1_5x"]
                and ep["sustained_success_step"] is not None
                and ep["sustained_success_step"] <= b
            )
            / n
            for b in budgets
        ]
        ax1.plot(budgets, ys, color=color, linestyle=ls, lw=2.0, label=label)

    ax1.axvline(220, color="#7f7f7f", linestyle=":", lw=1.0, alpha=0.7)
    ax1.axvline(300, color="#7f7f7f", linestyle=":", lw=1.0, alpha=0.7)
    ax1.axvline(520, color="#7f7f7f", linestyle=":", lw=1.0, alpha=0.7)
    ax1.text(224, 6, "spatial 1.0x", fontsize=7.2, color="#555555", rotation=90)
    ax1.text(304, 6, "goal 1.0x", fontsize=7.2, color="#555555", rotation=90)
    ax1.text(524, 6, "long 1.0x", fontsize=7.2, color="#555555", rotation=90)
    ax1.set_title("Cumulative Success Rate vs. Step Budget")
    ax1.set_xlabel("Environment Step Budget")
    ax1.set_ylabel("Cumulative Success Rate (%)")
    ax1.set_xlim(0, 750)
    ax1.set_ylim(0, 102)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(frameon=False, loc="lower right")

    tasks = ["spatial_t0", "object_t0", "goal_t0", "long_t0"]
    s50_id = _load_summary("il_act_50_id")["per_task"]
    s20_id = _load_summary("il_act_20_id")["per_task"]
    s50_ood = _load_summary("il_act_50_ood")["per_task"]
    s20_ood = _load_summary("il_act_20_ood")["per_task"]

    series = [
        ("il_act_50 ID", s50_id, "#1f77b4"),
        ("il_act_20 ID", s20_id, "#6baed6"),
        ("il_act_50 OOD", s50_ood, "#d62728"),
        ("il_act_20 OOD", s20_ood, "#fc9272"),
    ]
    x = np.arange(len(tasks))
    w = 0.19
    for idx, (label, pt, color) in enumerate(series):
        vals = []
        err_lo = []
        err_hi = []
        for t in tasks:
            k = pt[t]["sr_1_5x"]
            n = pt[t]["n"]
            p, lo, hi = wilson_ci_95(k, n)
            vals.append(p * 100.0)
            err_lo.append(max(0.0, (p - lo) * 100.0))
            err_hi.append(max(0.0, (hi - p) * 100.0))
        offset = (idx - 1.5) * w
        bars = ax2.bar(x + offset, vals, width=w, color=color, label=label, edgecolor="#222222", linewidth=0.4)
        ax2.errorbar(
            x + offset,
            vals,
            yerr=[err_lo, err_hi],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.9,
            capsize=2.0,
        )
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2.0, min(103.5, v + 1.5), f"{v:.0f}", ha="center", va="bottom", fontsize=6.8)

    ax2.set_title("Per-Task SR@1.5x Breakdown (95% Wilson CI)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["spatial_t0", "object_t0", "goal_t0", "long_t0"])
    ax2.set_ylabel("Success Rate @ 1.5x (%)")
    ax2.set_ylim(0, 126)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.legend(frameon=False, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_horizon_and_tasks.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig3_ood_axes_and_failures() -> None:
    s50_ood = _load_summary("il_act_50_ood")["per_ood_axis"]
    s20_ood = _load_summary("il_act_20_ood")["per_ood_axis"]
    axes_order = ["language_instruction", "objects_layout", "robot_init_state", "camera_visual"]
    axes_labels = ["Language\nParaphrase", "Objects\nLayout", "Robot Init\nState", "Camera &\nVisual"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.1))

    x = np.arange(len(axes_order))
    w = 0.34
    for idx, (label, src, color) in enumerate(
        [("il_act_50 OOD (79.5%)", s50_ood, "#1f77b4"), ("il_act_20 OOD (68.8%)", s20_ood, "#ff7f0e")]
    ):
        vals, lo_e, hi_e = [], [], []
        for ax_key in axes_order:
            p, lo, hi = wilson_ci_95(src[ax_key]["sr_1_5x"], src[ax_key]["n"])
            vals.append(p * 100.0)
            lo_e.append(max(0.0, (p - lo) * 100.0))
            hi_e.append(max(0.0, (hi - p) * 100.0))
        offset = (idx - 0.5) * w
        bars = ax1.bar(x + offset, vals, width=w, color=color, label=label, edgecolor="#222222", linewidth=0.5)
        ax1.errorbar(x + offset, vals, yerr=[lo_e, hi_e], fmt="none", ecolor="#333333", elinewidth=1.0, capsize=2.5)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2.0, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax1.set_title("TEST_OOD 4-Axis Robustness (McNemar p=0.0169)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(axes_labels)
    ax1.set_ylabel("Success Rate @ 1.5x (%)")
    ax1.set_ylim(0, 114)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.legend(frameon=False, loc="upper right")

    run_tags = [
        ("il_act_50_dev", "50 DEV"),
        ("il_act_20_dev", "20 DEV"),
        ("il_act_50_id", "50 ID"),
        ("il_act_20_id", "20 ID"),
        ("il_act_50_ood", "50 OOD"),
        ("il_act_20_ood", "20 OOD"),
    ]
    cluster_colors = {
        "success": "#2e7d32",
        "approach_miss": "#d9534f",
        "grasp_slip": "#f0ad4e",
        "place_or_articulate_fail": "#5bc0de",
        "timeout_incomplete": "#7b68ee",
    }
    cluster_labels = {
        "success": "Success",
        "approach_miss": "Approach Miss",
        "grasp_slip": "Grasp Slip",
        "place_or_articulate_fail": "Place/Artic. Fail",
        "timeout_incomplete": "Timeout Incomp.",
    }
    rx = np.arange(len(run_tags))
    bottoms = np.zeros(len(run_tags))
    counts_by_cluster: dict[str, list[float]] = {c: [] for c in CLUSTERS_ORDER}
    for rname, _ in run_tags:
        eps = _load_jsonl(ROOT_DIR / "runs" / rname / "episodes.jsonl")
        cnt = Counter(e["failure_cluster"] for e in eps)
        n = max(1, len(eps))
        for c in CLUSTERS_ORDER:
            counts_by_cluster[c].append(100.0 * cnt[c] / n)

    for c in CLUSTERS_ORDER:
        vals = np.array(counts_by_cluster[c])
        ax2.bar(rx, vals, bottom=bottoms, width=0.58, color=cluster_colors[c], label=cluster_labels[c])
        bottoms += vals

    ax2.set_title("4-Cluster Failure Breakdown Across All 624 Episodes")
    ax2.set_xticks(rx)
    ax2.set_xticklabels([lbl for _, lbl in run_tags], rotation=15)
    ax2.set_ylabel("Episode Share (%)")
    ax2.set_ylim(0, 114)
    ax2.legend(frameon=False, ncol=3, loc="upper center", fontsize=7.4)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_ood_axes_and_failures.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig4_cost_pareto() -> None:
    s50_id = _load_summary("il_act_50_id")
    s20_id = _load_summary("il_act_20_id")
    s50_ood = _load_summary("il_act_50_ood")
    s20_ood = _load_summary("il_act_20_ood")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7))

    pts = [
        ("il_act_50 TEST_ID", s50_id["cost_ledger"]["demo_count"], s50_id["cost_ledger"]["gpu_hours"], s50_id["sr_1_5x"] * 100, "#1f77b4", "o"),
        ("il_act_20 TEST_ID", s20_id["cost_ledger"]["demo_count"], s20_id["cost_ledger"]["gpu_hours"], s20_id["sr_1_5x"] * 100, "#1f77b4", "s"),
        ("il_act_50 TEST_OOD", s50_ood["cost_ledger"]["demo_count"], s50_ood["cost_ledger"]["gpu_hours"], s50_ood["sr_1_5x"] * 100, "#d62728", "o"),
        ("il_act_20 TEST_OOD", s20_ood["cost_ledger"]["demo_count"], s20_ood["cost_ledger"]["gpu_hours"], s20_ood["sr_1_5x"] * 100, "#d62728", "s"),
    ]

    for lbl, demos, gpuh, sr, col, mk in pts:
        ax1.scatter([demos], [sr], color=col, marker=mk, s=90, zorder=4)
        ax1.annotate(f"{lbl}\n({sr:.1f}%)", (demos, sr), textcoords="offset points", xytext=(-18 if demos > 120 else 8, -22 if "ID" in lbl and demos < 100 else 6), fontsize=7.8)
        ax2.scatter([gpuh], [sr], color=col, marker=mk, s=90, zorder=4)
        ax2.annotate(f"{lbl}\n({sr:.1f}%)", (gpuh, sr), textcoords="offset points", xytext=(-35 if gpuh > 0.7 else 8, -22 if "ID" in lbl and gpuh < 0.7 else 6), fontsize=7.8)

    ax1.plot([80, 165], [s20_id["sr_1_5x"] * 100, s50_id["sr_1_5x"] * 100], color="#1f77b4", lw=1.5, ls="--", alpha=0.6)
    ax1.plot([80, 165], [s20_ood["sr_1_5x"] * 100, s50_ood["sr_1_5x"] * 100], color="#d62728", lw=1.5, ls="--", alpha=0.6)
    ax1.set_title("Performance vs. Offline Demo Count")
    ax1.set_xlabel("Offline Expert Demonstrations (Episodes)")
    ax1.set_ylabel("Success Rate @ 1.5x (%)")
    ax1.set_xlim(60, 195)
    ax1.set_ylim(58, 99)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.plot([0.485, 0.901], [s20_id["sr_1_5x"] * 100, s50_id["sr_1_5x"] * 100], color="#1f77b4", lw=1.5, ls="--", alpha=0.6)
    ax2.plot([0.485, 0.901], [s20_ood["sr_1_5x"] * 100, s50_ood["sr_1_5x"] * 100], color="#d62728", lw=1.5, ls="--", alpha=0.6)
    ax2.set_title("Performance vs. RTX 5090 Training GPU Hours")
    ax2.set_xlabel("Training Compute (GPU Hours, RTX 5090)")
    ax2.set_ylabel("Success Rate @ 1.5x (%)")
    ax2.set_xlim(0.38, 1.05)
    ax2.set_ylim(58, 99)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_cost_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def stage_filmstrips() -> None:
    for split_dir in ("visual_showcase_id", "visual_showcase_ood"):
        src_dir = ROOT_DIR / "runs" / split_dir / "filmstrips"
        for png in sorted(src_dir.glob("*.png")):
            shutil.copy2(png, FIG_DIR / png.name)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_fig1_training_curves()
    make_fig2_horizon_and_tasks()
    make_fig3_ood_axes_and_failures()
    make_fig4_cost_pareto()
    stage_filmstrips()
    print("[OK] Generated PDF figures and staged filmstrips in:", FIG_DIR)
    for p in sorted(FIG_DIR.iterdir()):
        print(f"  - {p.name} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
