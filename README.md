# `libero-rsi-bench`

面向 **AIR5023 Course Project** 的统一具身操作评测基准与解耦评测平台（基于 `LIBERO` + `sylvestf/LIBERO-plus`）。本仓库专注**基准任务、数据切分锁、统一观测/动作接口、预算守卫与统计出图**，不限制组员各自使用的模型架构或训练框架。

---

## 核心使用逻辑（先看这 3 句）

1. **评测底座环境（GPU 服务器上配 1 套即可）**：Python 3.10 + `sylvestf/LIBERO-plus` 仿真器，负责运行 `evaluate.py` 与 `DevInteractionEnv`。
2. **组员策略环境（各自独立，不需要装 `LIBERO`）**：无论你用 `LeRobot`、`PyTorch`、`vLLM` 还是调闭源大模型 API，都留在自己的虚拟环境里，套一个 `HTTP Policy Server`（几行代码模板）连上底座端口即可评测，**彻底避免 Python 包冲突**。
3. **本地无 GPU（如 macOS）也能先调接口**：加上 `--env-backend mock` 即可零第三方依赖跑通全套协议与测试。

---

## 目录结构

```text
libero-rsi-bench/
├── docs/
│   ├── BENCHMARK_SPEC.md          # 小组实验基准与评测协议（任务、清洗版数据下载、训练预算、评测指标）
│   ├── INTEGRATION_GUIDE.md       # 组员接入规范（obs/action 格式、6 种 control_mode、HTTP 解耦、CaP）
│   └── videos/                    # 4 个核心任务的真机 MuJoCo 双视角成功演示视频（MP4）
├── protocol/
│   ├── tasks_manifest.json        # 4 核心任务 + 4 条 LIBERO-Plus OOD 轴 + 预算上限
│   ├── generate_splits.py         # 确定性切分生成与 SHA-256 锁校验脚本
│   └── splits.lock.json           # 已锁定切分：burned(8) / dev(100) / sealed_id(100) / sealed_ood(112)
├── harness/
│   ├── math_kinematics.py         # 四元数/轴角/旋转误差转换、SE(3) 像素反投影、确定性高斯噪声
│   ├── video_recorder.py          # 双视角画面拼接、零依赖 PNG 关键帧编码与 H.264 MP4 录制
│   ├── budget_tracker.py          # 2,000 回合硬预算守卫、sealed_reads.log 审计、4D 成本账本
│   ├── policy_protocol.py         # 跨环境 HTTP 协议与 Code-as-Policies 命令式桥接（ImperativePolicyBase）
│   ├── env_adapter.py             # sylvestf/LIBERO-plus 多控制模式实机适配器 + 纯 Python Mock 仿真器
│   ├── dev_env.py                 # 供 RL / RSI 使用的 Dev-only Gym 交互包装器（内置预算锁）
│   ├── evaluate.py                # 统一权威评测入口 CLI
│   └── stats_and_report.py        # Wilson 95% CI、精确 McNemar、4 类失败归因、Markdown & SVG 报表
├── examples/
│   ├── minimal_policy_server.py   # 常规方法接入 HTTP Policy Server 的最小模板
│   ├── minimal_code_policy.py     # Code-as-Policies（命令式脚本 + 闭环原语）HTTP 模板
│   └── minimal_dev_interaction.py # 在预算锁内使用 DevInteractionEnv 的最小模板
├── scripts/
│   └── setup_libero_plus.sh       # Linux GPU 评测环境一键安装脚本（克隆源码 + 装依赖 + 下资源 + 写配置）
├── tests/
│   └── verify_all.py              # 零第三方依赖全链路验证测试套件
├── requirements-harness.txt       # 评测底座 Python 依赖（LIBERO-plus 源码需配合 setup_libero_plus.sh 安装）
└── runs/                          # 评测结果、sealed_reads.log 与自动生成报表（不入 Git）
```

---

## 从零上手全流程（4 步）

### 第 1 步：安装 `LIBERO-plus` 评测环境（跑真机仿真 `libero` 后端时需要）

> **为什么 `requirements-harness.txt` 里没有直接写 `pip install libero`？**  
> 本基准使用的是支持 7 维 OOD 扰动的 **[sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)**（包名也叫 `libero`），它**未发布到 PyPI**，且运行仿真还需要单独下载 HuggingFace 上的场景贴图模型包 `assets.zip` 并生成配置文件。因此无法靠一条 `pip install -r` 搞定，请使用下面的一键脚本。

#### 方式 1：一键安装脚本（推荐，Linux / Ubuntu GPU 机）
```bash
conda create -n libero-bench python=3.10 -y && conda activate libero-bench
# Ubuntu 若尚未安装 ImageMagick / OpenGL / EGL 系统库，请先执行一次：
# sudo apt-get update && sudo apt-get install -y libmagickwand-dev libgl1 libegl1

bash scripts/setup_libero_plus.sh
```
脚本会自动：卸载旧版 `libero` $\rightarrow$ 克隆 `sylvestf/LIBERO-plus` 到 `third_party/LIBERO-plus` $\rightarrow$ 安装依赖 $\rightarrow$ 从 HuggingFace `Sylvest/LIBERO-plus` 下载解压 `assets.zip` $\rightarrow$ 写入非交互式 `~/.libero/config.yaml` 并自检 `OffScreenRenderEnv`。

#### 方式 2：已有 `LIBERO-plus` 环境或手动安装
若机器上已有现成的 `LIBERO-plus`（先移除原版 `libero`）：
```bash
git clone --depth 1 https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus
python -m pip install -r requirements-harness.txt -r third_party/LIBERO-plus/extra_requirements.txt
python -m pip install -e third_party/LIBERO-plus
# 从 https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main 下载 assets.zip 解压到 third_party/LIBERO-plus/libero/libero/
# 确保 ~/.libero/config.yaml（或 $LIBERO_CONFIG_PATH/config.yaml）中 assets / bddl_files / init_states 指向该目录
```

> **仅本地无 GPU 冒烟测试**：如果不装 `LIBERO-plus`，直接在 Mac 上也能运行零依赖自检：
> ```bash
> python3 protocol/generate_splits.py --verify && python3 tests/verify_all.py
> ```

---

### 第 2 步：各组员按方法类型接入（离线数据下载见 [BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md)）

* **模仿学习（IL）/ 常规 Agentic / RL 评测期**：参考 [examples/minimal_policy_server.py](examples/minimal_policy_server.py)，继承 `BasePolicy`，在 `reset()` 和 `act(obs_dict)` 里填入你的模型推理逻辑；可通过类属性 `control_mode` 自选 `osc_delta_eef` / `abs_eef` / `abs_joint_pos` / `macro_rel_eef` / `macro_target_eef`。
* **Code-as-Policies / 代码生成式 RSI**：参考 [examples/minimal_code_policy.py](examples/minimal_code_policy.py)，继承 `ImperativePolicyBase`，直接对大模型生成的 Python 控制代码调用 `api.move_to` / `api.move_delta` / `api.get_obs` 与 `if / while` 闭环执行。
* **RL 在线训练 / RSI 在 `dev` 上试错进化**：参考 [examples/minimal_dev_interaction.py](examples/minimal_dev_interaction.py)，使用 [harness/dev_env.py](harness/dev_env.py) 的 `DevInteractionEnv`，环境已锁定仅能采样 `dev` 切分且硬上限 `2,000` 回合。

---

### 第 3 步：日常开发调通与调参（`burned` $\rightarrow$ `dev`）

在**两个终端**分别运行策略服务和统一评测器（无头 Linux 服务器必须带 `MUJOCO_GL=egl`）：

```bash
# 终端 1（组员自己的训练/推理 Conda 环境，无需装 LIBERO）：启动 HTTP 策略服务
PYTHONPATH=. python3 your_policy_server.py --port 8000

# 终端 2（第 1 步装好的 libero-bench 环境）：先用 burned 8 题冒烟调通接口
MUJOCO_GL=egl python3 harness/evaluate.py \
  --split burned --env-backend libero --policy-url http://127.0.0.1:8000 --out runs/smoke_libero

# 接口通了以后，在 dev 100 题上开发调试、挑权重 / 迭代 RSI（可加 --save-videos 保存双视角视频）
MUJOCO_GL=egl python3 harness/evaluate.py \
  --split dev --env-backend libero --policy-url http://127.0.0.1:8000 --save-videos --out runs/my_method_dev
```

---

### 第 4 步：全组定稿后一次性跑最终测试并汇总出图（`sealed_id` + `sealed_ood`）

各方法定稿冻结后，统一用同一套命令打开密封测试集（需加 `--confirm-unseal`）并生成对比报告与矢量图：

```bash
# 1. 跑域内密封测试（100 题）与域外抗干扰密封测试（112 题）
MUJOCO_GL=egl python3 harness/evaluate.py --split sealed_id  --env-backend libero --confirm-unseal --policy-url http://127.0.0.1:8000 --demo-count 200 --gpu-hours 4.0 --out runs/sealed_id_my_method
MUJOCO_GL=egl python3 harness/evaluate.py --split sealed_ood --env-backend libero --confirm-unseal --policy-url http://127.0.0.1:8000 --demo-count 200 --gpu-hours 4.0 --out runs/sealed_ood_my_method

# 2. 汇总全部组员结果生成 Markdown 对比表 + 3 张 SVG 图（horizon / failure_clusters / cost_pareto）
python3 harness/stats_and_report.py --runs runs/sealed_id_* --out runs/final_report_id
python3 harness/stats_and_report.py --runs runs/sealed_ood_* --out runs/final_report_ood
```

详细任务、数据筛选、预算规则见 [docs/BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md)，全字段接口文档见 [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)。
