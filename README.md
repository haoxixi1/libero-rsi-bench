# `libero-rsi-bench`

面向 **AIR5023 Course Project** 的统一具身操作评测基准与解耦评测平台。本仓库专注于**基准定义、切分与防偷看锁、统一观测与步数契约、在线交互预算守卫、HTTP 跨环境通信协议以及统计学汇总出图**，不绑定或预设各成员使用的具体算法与训练框架。

---

## 目录结构

```text
libero-rsi-bench/
├── docs/
│   ├── BENCHMARK_SPEC.md          # 小组实验基准与评测协议（任务、数据下载、训练预算、评测指标）
│   ├── INTEGRATION_GUIDE.md       # 组员接入规范（obs/action 格式、HTTP Server 解耦、DevInteractionEnv）
│   └── videos/                    # 4 个核心任务的真机 MuJoCo 双视角成功演示视频（MP4）
├── protocol/
│   ├── tasks_manifest.json        # 4 核心任务（含 clean_base_name）+ 4 条 LIBERO-Plus OOD 轴 + 预算上限
│   ├── generate_splits.py         # 确定性切分生成与 SHA-256 锁校验脚本
│   └── splits.lock.json           # 已锁定切分：burned(8) / dev(100) / sealed_id(100) / sealed_ood(112)
├── harness/
│   ├── math_kinematics.py         # 四元数/轴角/旋转误差转换、SE(3) 像素反投影、确定性高斯噪声
│   ├── video_recorder.py          # 双视角画面拼接、零依赖 PNG 关键帧编码与 H.264 MP4 录制
│   ├── budget_tracker.py          # 2,000 回合硬预算守卫、sealed_reads.log 审计、4D 成本账本
│   ├── policy_protocol.py         # 零依赖跨环境协议（control_mode 协商 / PolicyHTTPServer / Client）
│   ├── env_adapter.py             # sylvestf/LIBERO-plus 多控制模式实机适配器 + 纯 Python 物理 Mock 仿真器
│   ├── dev_env.py                 # 供 RL / RSI 使用的 Dev-only Gym 交互包装器（内置预算锁）
│   ├── evaluate.py                # 统一权威评测入口 CLI
│   └── stats_and_report.py        # Wilson 95% CI、精确 McNemar、有限样本功效、Markdown & SVG 报表
├── examples/
│   ├── minimal_policy_server.py   # 任意方法接入 HTTP Policy Server 的最小模板
│   ├── minimal_code_policy.py     # Code-as-Policies（命令式脚本 + 闭环原语）HTTP 模板
│   └── minimal_dev_interaction.py # 在预算锁内使用 DevInteractionEnv 的最小模板
├── tests/
│   └── verify_all.py              # 零第三方依赖全链路验证测试套件
├── requirements-harness.txt       # 评测底座环境依赖（LIBERO-plus + MuJoCo 2.3.7）
└── runs/                          # 评测结果、sealed_reads.log 与自动生成报表（不入 Git）
```

---

## 快速验证（本地无 GPU / 服务器实机）

### 1. 本地零第三方依赖 Mock 验证
```bash
python3 protocol/generate_splits.py --verify
python3 tests/verify_all.py
```

### 2. 在 `LIBERO-plus` GPU 服务器上联调接入
```bash
# 终端 1：在你的算法环境中启动 HTTP 策略服务（参考 examples/minimal_policy_server.py）
python3 examples/minimal_policy_server.py --family il --port 8000

# 终端 2：在 LIBERO-plus 环境中运行权威评测器（无头环境需 MUJOCO_GL=egl）
MUJOCO_GL=egl python3 harness/evaluate.py \
  --split burned \
  --env-backend libero \
  --policy-url http://127.0.0.1:8000 \
  --out runs/smoke_libero
```

详细协议与接口定义见 [docs/BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md) 与 [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)。
