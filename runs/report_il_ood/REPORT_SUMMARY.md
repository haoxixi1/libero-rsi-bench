# Benchmark Comparison Summary (`libero-rsi-bench`)

## 1. Main Results & Lifecycle Cost

| Run | Family | N | SR@1.0x | SR@1.5x (95% Wilson CI) | Mean Steps | Mean Macro Turns | Online Eps | Total Tokens |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `il_act_50_ood` | `il` | 112 | 78.6% | 79.5% [71.1%, 85.9%] | 241.2 | 0.00 | 112 | 0 |
| `il_act_20_ood` | `il` | 112 | 67.9% | 68.8% [59.7%, 76.6%] | 276.2 | 0.00 | 112 | 0 |

## 2. 4-Cluster Failure Breakdown

| Run | success | approach_miss | grasp_slip | place_or_articulate_fail | timeout_incomplete |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `il_act_50_ood` | 89 | 8 | 4 | 7 | 4 |
| `il_act_20_ood` | 77 | 15 | 4 | 11 | 5 |

## 3. Paired Exact McNemar Tests (SR@1.5x)

| Method A | Method B | Shared N | A Wins | B Wins | Exact Two-Sided p |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `il_act_50_ood` | `il_act_20_ood` | 112 | 17 | 5 | 0.0169 |

## 4. Generated Figures
- `horizon_curve.svg`
- `failure_clusters.svg`
- `cost_pareto.svg`
