# `libero-rsi-bench`

面向 **AIR5023 Course Project** 的统一具身操作评测基准（基于 `LIBERO` + `sylvestf/LIBERO-plus`）。每个组员直接在**自己的 GPU 机器**上 `git clone` 本仓库接入各自方法并跑分，最后把生成的 `runs/` 结果文件夹汇总到一起自动生成全组对比表和图表即可。

---

## 核心使用逻辑（先看这 3 句）

1. **各自独立运行**：这就是一套公共 Benchmark 代码和考题集，每位组员都在自己机器上用同一套评测脚本跑出成绩。
2. **怎么接你的方法（任选一种）**：
   * 如果你的模型环境与 `LIBERO-plus` 兼容：直接**单条命令**用 `--policy-class 你的模块:你的策略类` 开跑；
   * 如果有依赖冲突（比如新版 `LeRobot` / `PyTorch` 与 `mujoco 2.3.7` 冲突）：在本机开一个本地端口起 `HTTP Policy Server` 连过去即可，**完全不用折腾合并环境**。
3. **平时在 `dev` 调试，最后跑 `test_id` + `test_ood` 交结果**：定稿后把 `runs/<method>_id` 和 `runs/<method>_ood` 目录发给组长合并出图。

---

## 目录结构

```text
libero-rsi-bench/
├── docs/
│   ├── BENCHMARK_SPEC.md          # 任务清单、清洗版数据集下载、训练预算与评测口径
│   ├── INTEGRATION_GUIDE.md       # 观测/动作格式、6 种 control_mode、HTTP 解耦与 Code-as-Policies 接入
│   └── videos/                    # 4 个核心任务的真机 MuJoCo 双视角成功演示视频（MP4）
├── protocol/
│   ├── tasks_manifest.json        # 4 核心任务 + 4 条 LIBERO-Plus OOD 轴 + 预算上限
│   ├── generate_splits.py         # 确定性切分与校验脚本
│   └── splits.lock.json           # 已锁定切分：dev(100) / test_id(100) / test_ood(112)
├── harness/
│   ├── math_kinematics.py         # 四元数/轴角/旋转误差转换、SE(3) 像素反投影
│   ├── video_recorder.py          # 双视角拼接、PNG 关键帧与 H.264 MP4 录制
│   ├── budget_tracker.py          # 2,000 回合 Dev 在线预算守卫与 4D 成本账本
│   ├── policy_protocol.py         # HTTP 跨环境协议与 Code-as-Policies 桥接（ImperativePolicyBase）
│   ├── env_adapter.py             # sylvestf/LIBERO-plus 多控制模式适配器 + 纯 Python Mock 仿真器
│   ├── dev_env.py                 # 供 RL / RSI 使用的 Dev-only Gym 交互包装器（内置预算锁）
│   ├── evaluate.py                # 统一评测入口 CLI
│   └── stats_and_report.py        # 汇总各组员 runs/ 生成 Markdown 对比表 + 3 张 SVG 图
├── examples/
│   ├── minimal_policy_server.py   # 常规方法接入模板
│   ├── minimal_code_policy.py     # Code-as-Policies（命令式脚本 + 闭环原语）接入模板
│   └── minimal_dev_interaction.py # 在线 RL / RSI 交互环境模板
├── scripts/
│   └── setup_libero_plus.sh       # Linux GPU 评测环境一键安装脚本（克隆源码 + 装依赖 + 下资源 + 写配置）
├── tests/
│   └── verify_all.py              # 本地零依赖全链路测试套件
├── requirements-harness.txt       # 评测环境 Python 依赖
└── runs/                          # 评测输出（summary.json / episodes.jsonl / 可选视频）
```

---

## 组员上手 3 步走

### 第 1 步：在你的 GPU 机器上安装 `LIBERO-plus` 评测环境

> **说明**：本基准使用的是支持 OOD 扰动的 **[sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)**（未发布在 PyPI，且需从 HuggingFace 拉取 `assets.zip`），请用仓库内置的一键脚本安装（需 **Python 3.10**）：

```bash
git clone https://github.com/haoxixi1/libero-rsi-bench.git && cd libero-rsi-bench
conda create -n libero-bench python=3.10 -y && conda activate libero-bench
# Ubuntu 若尚未安装 ImageMagick / OpenGL / EGL 系统库，先运行一次：
# sudo apt-get update && sudo apt-get install -y libmagickwand-dev libgl1 libegl1

bash scripts/setup_libero_plus.sh
```
*(如果机器上只跑轻量本地接口自检，不装 `LIBERO-plus` 也可直接运行 `python3 tests/verify_all.py` 或 `--env-backend mock`。)*

---

### 第 2 步：接入你的方法并在 `dev` 集调试（数据下载与预算上限见 [BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md)）

* **接策略接口**：
  * **IL / RL 评测 / Agentic**：继承 `BasePolicy`，写好 `reset()` 与 `act(obs_dict)`（参考 [examples/minimal_policy_server.py](examples/minimal_policy_server.py)，可自选 `osc_delta_eef` / `abs_eef` / `abs_joint_pos` / `macro_rel_eef` / `macro_target_eef`）。
  * **Code-as-Policies（写控制代码）**：继承 `ImperativePolicyBase`（参考 [examples/minimal_code_policy.py](examples/minimal_code_policy.py)）。
  * **RL 在线训练 / RSI 迭代试错**：用 `DevInteractionEnv`（参考 [examples/minimal_dev_interaction.py](examples/minimal_dev_interaction.py)），自动锁定仅访问 `dev` 且不超 2,000 回合上限。
* **在本机运行评测调试**：
  ```bash
  # 方式 1：同环境直接单命令跑（无包冲突时最省事）
  MUJOCO_GL=egl python3 harness/evaluate.py --split dev --env-backend libero --policy-class my_pkg.my_policy:MyPolicy --limit 4 --out runs/debug

  # 方式 2：如果算法环境与 LIBERO-plus 冲突，在自己本机起一个端口解耦跑
  #   终端 A（算法环境）: PYTHONPATH=. python3 your_policy_server.py --port 8000
  #   终端 B（libero-bench）:
  MUJOCO_GL=egl python3 harness/evaluate.py --split dev --env-backend libero --policy-url http://127.0.0.1:8000 --out runs/my_method_dev
  ```
  平时改完代码先加 `--limit 4` 秒测通不通，调参或选最佳权重时跑完整 `--split dev`（100 题，可加 `--save-videos` 看回放）。

---

### 第 3 步：在本机跑完 `test_id` / `test_ood` 并收集结果出图

1. **每位组员**方法定稿后，直接在**自己的机器**上跑最终测试集（域内 100 题 + 域外抗干扰 112 题）：
   ```bash
   MUJOCO_GL=egl python3 harness/evaluate.py --split test_id  --env-backend libero --policy-url http://127.0.0.1:8000 --demo-count 200 --gpu-hours 4.0 --out runs/my_method_id
   MUJOCO_GL=egl python3 harness/evaluate.py --split test_ood --env-backend libero --policy-url http://127.0.0.1:8000 --demo-count 200 --gpu-hours 4.0 --out runs/my_method_ood
   ```
   *(若使用 `--policy-class` 同理，把 `--policy-url ...` 替换掉即可；跑完后会生成包含 `summary.json` 与 `episodes.jsonl` 的结果目录。)*
2. **汇总出图**：把大家各自跑出来的 `runs/` 结果文件夹汇总到一个目录，一键生成全组对比表与 3 张矢量图：
   ```bash
   python3 harness/stats_and_report.py --runs runs/*_id  --out runs/final_report_id
   python3 harness/stats_and_report.py --runs runs/*_ood --out runs/final_report_ood
   ```

完整协议细节见 [docs/BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md) 与 [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)。
