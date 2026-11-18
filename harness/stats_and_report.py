"""Pure-Python exact statistical tests (Wilson CI, Exact McNemar, Power) & SVG report generator."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CLUSTERS_ORDER = (
    "success",
    "approach_miss",
    "grasp_slip",
    "place_or_articulate_fail",
    "timeout_incomplete",
)

CLUSTER_COLORS = {
    "success": "#2e7d32",
    "approach_miss": "#d9534f",
    "grasp_slip": "#f0ad4e",
    "place_or_articulate_fail": "#5bc0de",
    "timeout_incomplete": "#7b68ee",
}

SERIES_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def wilson_ci_95(k: int, n: int) -> tuple[float, float, float]:
    """Return (proportion, lower_95, upper_95) using the Wilson score interval."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054
    z2 = z * z
    p = k / n
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _binom_pmf_half(m: int, i: int) -> float:
    """Exact Binomial(m, 0.5) PMF computed safely in log-space."""
    log_p = math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1) - m * math.log(2.0)
    return math.exp(log_p)


def exact_mcnemar_test(wins_a: int, wins_b: int, *, alternative: str = "two-sided") -> float:
    """Compute exact McNemar p-value comparing candidate A (wins_a) vs baseline B (wins_b)."""
    wins_a = int(wins_a)
    wins_b = int(wins_b)
    m = wins_a + wins_b
    if m == 0:
        return 1.0
    if alternative == "greater":
        return min(1.0, sum(_binom_pmf_half(m, i) for i in range(wins_a, m + 1)))
    if alternative == "two-sided":
        k_min = min(wins_a, wins_b)
        tail = sum(_binom_pmf_half(m, i) for i in range(0, k_min + 1))
        return min(1.0, 2.0 * tail)
    raise ValueError(f"Unsupported alternative: {alternative!r}")


def paired_counts(outcomes_a: Iterable[bool], outcomes_b: Iterable[bool]) -> tuple[int, int]:
    """Return (wins_a, wins_b) over paired boolean outcome sequences."""
    wins_a = 0
    wins_b = 0
    for a, b in zip(outcomes_a, outcomes_b):
        if bool(a) and not bool(b):
            wins_a += 1
        elif bool(b) and not bool(a):
            wins_b += 1
    return wins_a, wins_b


def mcnemar_power(
    n_total: int,
    delta_pt: float,
    *,
    discordant_base: float = 0.08,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Compute finite-sample statistical power of exact McNemar test under paired trinomial model."""
    delta = delta_pt / 100.0
    p10 = max(1e-6, float(discordant_base))
    p01 = max(1e-6, p10 + delta)
    if p10 + p01 >= 1.0:
        scale = 0.98 / (p10 + p01)
        p10 *= scale
        p01 *= scale
    p_tie = 1.0 - p10 - p01

    log_n_fact = math.lgamma(n_total + 1)
    log_p01 = math.log(p01)
    log_p10 = math.log(p10)
    log_ptie = math.log(max(p_tie, 1e-12))

    # Precompute minimal winning count threshold for each discordant total m
    power = 0.0
    for m in range(1, n_total + 1):
        ties = n_total - m
        for n01 in range(m + 1):
            n10 = m - n01
            if n01 <= n10:
                continue
            p_val = exact_mcnemar_test(n01, n10, alternative=alternative)
            if p_val <= alpha:
                log_prob = (
                    log_n_fact
                    - math.lgamma(n01 + 1)
                    - math.lgamma(n10 + 1)
                    - math.lgamma(ties + 1)
                    + n01 * log_p01
                    + n10 * log_p10
                    + ties * log_ptie
                )
                power += math.exp(log_prob)
    return min(1.0, power)


def _load_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    episodes_path = run_dir / "episodes.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes: list[dict[str, Any]] = []
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                episodes.append(json.loads(line))
    return {"dir": run_dir, "name": run_dir.name, "summary": summary, "episodes": episodes}


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_horizon_svg(runs: list[dict[str, Any]], out_path: Path) -> None:
    width, height = 760, 420
    pad_l, pad_r, pad_t, pad_b = 70, 190, 45, 55
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    horizons = list(range(50, 751, 50))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="380" y="25" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold">Success Rate vs. Max Allowed Env Steps</text>',
    ]
    # Axes and grid
    for pct in (0, 25, 50, 75, 100):
        y = pad_t + plot_h * (1.0 - pct / 100.0)
        lines.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{pct}%</text>')

    for h_val in (100, 200, 300, 450, 600, 750):
        x = pad_l + plot_w * (h_val - 50) / 700.0
        lines.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" stroke="#f0f0f0"/>')
        lines.append(f'<text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" font-family="sans-serif" font-size="11">{h_val}</text>')

    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333333"/>')
    lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 12}" text-anchor="middle" font-family="sans-serif" font-size="12">Max Env Steps Horizon</text>')

    for idx, run in enumerate(runs):
        color = SERIES_PALETTE[idx % len(SERIES_PALETTE)]
        eps = run["episodes"]
        n = max(1, len(eps))
        pts: list[str] = []
        for h_val in horizons:
            succ = sum(
                1
                for ep in eps
                if ep.get("sustained_success_step") is not None and int(ep["sustained_success_step"]) <= h_val
            )
            rate = 100.0 * succ / n
            x = pad_l + plot_w * (h_val - 50) / 700.0
            y = pad_t + plot_h * (1.0 - rate / 100.0)
            pts.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(pts)}"/>')
        ly = pad_t + 18 + idx * 22
        lx = pad_l + plot_w + 15
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 20}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{lx + 26}" y="{ly + 4}" font-family="sans-serif" font-size="11">{_svg_escape(run["name"])}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_failure_clusters_svg(runs: list[dict[str, Any]], out_path: Path) -> None:
    width, height = 760, 400
    pad_l, pad_r, pad_t, pad_b = 70, 190, 45, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="380" y="25" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold">Outcome &amp; 4-Cluster Failure Breakdown</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333333"/>',
    ]

    n_runs = max(1, len(runs))
    slot_w = plot_w / n_runs
    bar_w = min(56.0, slot_w * 0.6)

    for idx, run in enumerate(runs):
        eps = run["episodes"]
        total = max(1, len(eps))
        counts = Counter(ep.get("failure_cluster", "place_or_articulate_fail") for ep in eps)
        bx = pad_l + idx * slot_w + (slot_w - bar_w) / 2.0
        cur_y = pad_t + plot_h
        for cluster in CLUSTERS_ORDER:
            frac = counts.get(cluster, 0) / total
            bh = frac * plot_h
            if bh > 0.5:
                cur_y -= bh
                lines.append(
                    f'<rect x="{bx:.1f}" y="{cur_y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                    f'fill="{CLUSTER_COLORS[cluster]}" stroke="#ffffff" stroke-width="0.8"/>'
                )
        lines.append(
            f'<text x="{bx + bar_w / 2:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{_svg_escape(run["name"][:16])}</text>'
        )

    for idx, cluster in enumerate(CLUSTERS_ORDER):
        ly = pad_t + 16 + idx * 24
        lx = pad_l + plot_w + 15
        lines.append(f'<rect x="{lx}" y="{ly - 10}" width="14" height="14" fill="{CLUSTER_COLORS[cluster]}"/>')
        lines.append(f'<text x="{lx + 20}" y="{ly + 1}" font-family="sans-serif" font-size="11">{cluster}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_cost_pareto_svg(runs: list[dict[str, Any]], out_path: Path) -> None:
    width, height = 760, 400
    pad_l, pad_r, pad_t, pad_b = 75, 45, 45, 55
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    max_tokens = max(
        [1000]
        + [int(r["summary"].get("cost_ledger", {}).get("total_tokens", 0)) for r in runs]
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="380" y="25" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold">Performance vs. Token Cost (SR@1.5x)</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333333"/>',
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 12}" text-anchor="middle" font-family="sans-serif" font-size="12">Total LLM Tokens (Eval + Optimization)</text>',
    ]
    for pct in (0, 25, 50, 75, 100):
        y = pad_t + plot_h * (1.0 - pct / 100.0)
        lines.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{pct}%</text>')

    for idx, run in enumerate(runs):
        color = SERIES_PALETTE[idx % len(SERIES_PALETTE)]
        summary = run["summary"]
        sr_1_5 = 100.0 * float(summary.get("sr_1_5x", 0.0))
        tokens = float(summary.get("cost_ledger", {}).get("total_tokens", 0))
        x = pad_l + plot_w * (math.log10(1.0 + tokens) / math.log10(1.0 + max_tokens))
        y = pad_t + plot_h * (1.0 - sr_1_5 / 100.0)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" opacity="0.85"/>')
        lines.append(
            f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" font-family="sans-serif" font-size="11">'
            f'{_svg_escape(run["name"])} ({sr_1_5:.1f}%)</text>'
        )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_comparison_report(run_dirs: list[Path], out_dir: Path) -> Path:
    """Aggregate multiple run directories into REPORT_SUMMARY.md and 3 SVG plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(Path(d)) for d in run_dirs]

    _write_horizon_svg(runs, out_dir / "horizon_curve.svg")
    _write_failure_clusters_svg(runs, out_dir / "failure_clusters.svg")
    _write_cost_pareto_svg(runs, out_dir / "cost_pareto.svg")

    md: list[str] = [
        "# Benchmark Comparison Summary (`libero-rsi-bench`)",
        "",
        "## 1. Main Results & Lifecycle Cost",
        "",
        "| Run | Family | N | SR@1.0x | SR@1.5x (95% Wilson CI) | Mean Steps | Mean Macro Turns | Online Eps | Total Tokens |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in runs:
        s = r["summary"]
        n = int(s.get("num_episodes", len(r["episodes"])))
        k1 = int(s.get("success_1_0x_count", 0))
        k2 = int(s.get("success_1_5x_count", 0))
        p1, _, _ = wilson_ci_95(k1, n)
        p2, lo2, hi2 = wilson_ci_95(k2, n)
        ledger = s.get("cost_ledger", {})
        md.append(
            f"| `{r['name']}` | `{s.get('family', '?')}` | {n} | "
            f"{p1 * 100:.1f}% | {p2 * 100:.1f}% [{lo2 * 100:.1f}%, {hi2 * 100:.1f}%] | "
            f"{s.get('mean_steps', 0.0):.1f} | {s.get('mean_macro_turns', 0.0):.2f} | "
            f"{ledger.get('online_episodes', 0)} | {ledger.get('total_tokens', 0)} |"
        )

    md.extend(["", "## 2. 4-Cluster Failure Breakdown", ""])
    md.append("| Run | success | approach_miss | grasp_slip | place_or_articulate_fail | timeout_incomplete |")
    md.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for r in runs:
        counts = Counter(ep.get("failure_cluster", "place_or_articulate_fail") for ep in r["episodes"])
        md.append(
            f"| `{r['name']}` | "
            + " | ".join(str(counts.get(c, 0)) for c in CLUSTERS_ORDER)
            + " |"
        )

    if len(runs) >= 2:
        md.extend(["", "## 3. Paired Exact McNemar Tests (SR@1.5x)", ""])
        md.append("| Method A | Method B | Shared N | A Wins | B Wins | Exact Two-Sided p |")
        md.append("| :--- | :--- | ---: | ---: | ---: | ---: |")
        ep_maps = [
            {ep["episode_id"]: bool(ep["success_1_5x"]) for ep in r["episodes"]}
            for r in runs
        ]
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                shared = sorted(set(ep_maps[i]) & set(ep_maps[j]))
                if not shared:
                    continue
                wa, wb = paired_counts(
                    [ep_maps[i][k] for k in shared],
                    [ep_maps[j][k] for k in shared],
                )
                p_val = exact_mcnemar_test(wa, wb, alternative="two-sided")
                md.append(
                    f"| `{runs[i]['name']}` | `{runs[j]['name']}` | {len(shared)} | {wa} | {wb} | {p_val:.4f} |"
                )

    md.extend(
        [
            "",
            "## 4. Generated Figures",
            "- `horizon_curve.svg`",
            "- `failure_clusters.svg`",
            "- `cost_pareto.svg`",
            "",
        ]
    )
    report_path = out_dir / "REPORT_SUMMARY.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run outputs and compute statistical reports.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Paths to run directories")
    parser.add_argument("--out", type=Path, required=True, help="Output report directory")
    args = parser.parse_args()
    report_path = generate_comparison_report(args.runs, args.out)
    print(f"Generated report at {report_path}")


if __name__ == "__main__":
    main()
