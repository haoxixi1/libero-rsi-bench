# 组员接入与部署规范（Integration Guide）

无论你负责 **IL、RL、Agentic 还是 RSI**，也无论你使用 `LeRobot`、`starVLA`、`OpenVLA`、`PyTorch` 还是直接调用大模型 API，你只需将策略接入本仓库提供的两个标准接口之一，即可由统一的权威评测器打分与出图。

---

## 1. 观测字典与动作空间规范

### 评测器传给策略的 `obs_dict`
每次调用 `policy.act(obs_dict)` 时，你将收到经过脱敏处理的标准化字典：

| 字段 | 类型 / 形状 | 说明 |
| :--- | :--- | :--- |
| `obs_dict["images"]["agentview"]` | `uint8 (256, 256, 3)` | 第三人称全局 RGB（已做 180° 旋转对齐） |
| `obs_dict["images"]["wrist"]` | `uint8 (256, 256, 3)` | 机械臂腕部相机 RGB（已做 180° 旋转对齐） |
| `obs_dict["depth"]["agentview"]` / `["wrist"]` | `float32 (256, 256, 1)` | 物理米制深度（经 `get_real_depth_map` 转换，仅 `rgbd_projection` 下提供） |
| `obs_dict["camera_intrinsics"]` / `["camera_extrinsics"]` | `dict[str, list]` | 已按 `[::-1, ::-1]` 旋转图像校准的标称 3×3 内参与 4×4 外参矩阵（仅 `rgbd_projection` 下提供） |
| `obs_dict["state"]` | `list[float]` 长度 `8` | `[eef_x, eef_y, eef_z, axisangle_rx, ry, rz, gripper_q0, gripper_q1]` |
| `obs_dict["eef_pos"]` / `["eef_quat"]` | `list[3]` / `list[4]` | 末端世界坐标与四元数（`xyzw`） |
| `obs_dict["joint_pos"]` / `["joint_vel"]` | `list[7]` / `list[7]` | Franka Panda 7 自由度关节角（`rad`）与关节角速度（`rad/s`，纯本体感觉） |
| `obs_dict["gripper_qpos"]` | `list[2]` | 夹爪左右指节位置 |
| `obs_dict["instruction"]` | `str` | 当前回合语言指令（`language_instruction` 轴为改写句） |
| `obs_dict["task_key"]` | `str` | 当前任务标识（不暴露隐藏物理标签或物体真值坐标） |
| `obs_dict["step"]` / `["remaining_steps"]` | `int` | 当前物理步数与距离 `extended_steps` 的剩余步数 |
| `obs_dict["keypoints"]` | `dict` *(可选)* | 仅在 `--perception-mode noisy_keypoints` 消融轨提供（含 $\sigma=10\text{ mm}$ 确定性噪声） |

### 策略返回的动作、控制模式与成本口径
在你的 `BasePolicy` 子类中设置类属性 `control_mode`（`PolicyHTTPServer` 会通过 `GET /health` 自动上报给评测器，也可用 `evaluate.py --control-mode` 显式指定）。所有模式统一按底层 **20Hz（50ms）MuJoCo 物理步** 扣减预算：

| `control_mode` | 返回维度与含义 | 底层执行与计步口径 |
| :--- | :--- | :--- |
| **`osc_delta_eef`** *(默认，别名 `delta_eef`)* | `7D`: `[dx, dy, dz, drx, dry, drz, grip]` $\in [-1, 1]$ | `OSC_POSE (use_delta=True)` 单步直通（支持单步或 Chunk 序列；每步记 1 物理步） |
| **`abs_eef`** | `7D`: `[x, y, z, rx, ry, rz, grip]`（世界系米/轴角弧度） | `OSC_POSE (use_delta=False)` 单步直通（适合绝对位姿 Chunk；每步记 1 物理步） |
| **`abs_joint_pos`** | `8D`: `[q1..q7, grip]`（绝对关节角 `rad` + 夹爪 `±1`） | `JOINT_POSITION` 控制器单步跟踪 `clip((q_tgt - q_cur)/0.05, -1, 1)`（每步记 1 物理步） |
| **`delta_joint_pos`** | `8D`: `[dq1..dq7, grip]` $\in [-1, 1]$ | `JOINT_POSITION` 归一化增量单步直通（每步记 1 物理步） |
| **`macro_rel_eef`** | `7D` `[Δx, Δy, Δz, Δrx, Δry, Δrz, grip]` 或 `4D` `[Δx, Δy, Δz, grip]`（相对当前夹爪的米/弧度偏移） | 评测底座内置锁死闭环 P 伺服步进到位（最长 40 子步，误差 ≤ 1.2 cm；走 $k$ 子步记 $k$ 物理步） |
| **`macro_target_eef`** *(别名 `target_eef`)* | `7D` `[x, y, z, rx, ry, rz, grip]` 或 `4D` `[x, y, z, grip]`（世界系目标路点） | 与 `macro_rel_eef` 共用同一个锁死 P 伺服（走 $k$ 子步记 $k$ 物理步，命中即停） |

> **注**：夹爪维度统一约定 `> 0`（如 `+1.0`）为闭合，`<= 0`（如 `-1.0`）为张开；两个 `macro_*` 模式若发生夹爪开/合切换，伺服会自动至少保持 8 个物理步等待夹爪动作到位。

`policy.act(obs_dict)` 返回 `(action, call_cost)`：
* **`action`**：可以是单步向量 `[...]`，也可以是一个 **Action Chunk** `[[...], [...], ...]`（评测器逐帧/逐路点推进并在命中官方 `env.check_success()` 时立即终止）。
* **`call_cost`** *(字典，可为空)*：若有大模型调用，请返回：
  ```python
  {
      "macro_turn_increment": 1,  # 若本次触发了一次高层技能/规划决策
      "llm_unique_calls": 1,
      "llm_total_calls": 1,
      "prompt_tokens": 600,
      "completion_tokens": 50,
      "reasoning_tokens": 0,
  }
  ```

---

## 2. 两种本地运行与接入方式

每位组员在**自己的 GPU 机器**上按 [一键安装步骤](../README.md#第-1-步在你的-gpu-机器上安装-libero-plus-评测环境) 配好环境后，按是否与模型库存在 Python 包冲突二选一即可：

### 方式 A（无冲突最省事）：单命令直接加载策略类（`--policy-class`）
如果你的策略与 `LIBERO-plus` 在同一个 Python 环境中，直接一条命令开跑（开发调试可加 `--limit 4` 秒跑几题）：
```bash
MUJOCO_GL=egl python3 harness/evaluate.py \
  --split dev \
  --env-backend libero \
  --policy-class my_pkg.my_policy:MyPolicyClass \
  --out runs/my_method_dev
```

### 方式 B（推荐用于不同虚拟环境）：本机 HTTP Policy Server 解耦
如果你的训练环境（例如装有新版 `PyTorch`/`transformers`/`LeRobot`）与 `LIBERO-plus`（`mujoco==2.3.7`, `numpy<2.0`）存在 Python 包冲突，**完全不需要把环境合并**！直接在你自己的机器上开两个终端即可：

1. 在**算法环境终端**继承 `BasePolicy` 并用 `PolicyHTTPServer` 暴露本地端口（参考模板 [examples/minimal_policy_server.py](../examples/minimal_policy_server.py)）：
   ```bash
   PYTHONPATH=/path/to/libero-rsi-bench python3 your_policy_server.py --port 8001
   ```
2. 在 **`libero-bench` 环境终端**连上去跑分（无头服务器需设置 `MUJOCO_GL=egl`）：
   ```bash
   # 1. 快速抽 4 题冒烟调通接口
   MUJOCO_GL=egl python3 harness/evaluate.py --split dev --limit 4 --env-backend libero --policy-url http://127.0.0.1:8001 --out runs/smoke_test

   # 2. 在 dev 100 题上完整调试与选模（可加 --save-videos 或 --save-keyframes 导出双视角视频与关键帧）
   MUJOCO_GL=egl python3 harness/evaluate.py --split dev --env-backend libero --policy-url http://127.0.0.1:8001 --save-videos --out runs/my_method_dev
   ```

### 补充：Code as Policies（代码生成式策略）接入
如果组员做 Agentic / RSI 时希望大模型**直接生成一段包含 `if / while` 闭环逻辑的控制代码**来驱动机械臂，有两种接入方式：
1. **开环路点列表**：在 `act(obs_dict)` 里用代码算出整条路点序列 `[[x1, y1, z1, g1], [x2, y2, z2, g2], ...]` 一次返回给底座逐步伺服；
2. **闭环命令式脚本（`if / while` + `move_to` / `get_obs`）**：继承 [policy_protocol.py](../harness/policy_protocol.py) 里的 **`ImperativePolicyBase`**，在 `run_episode(self, api, task_meta)` 中把大模型生成的代码（含 `api.get_obs()`、`api.move_to(...)`、`api.move_delta(...)`、`api.open_gripper()` / `api.close_gripper()`、`api.record_llm_cost(...)`）直接跑在跨进程同步桥接上（参考模板 [examples/minimal_code_policy.py](../examples/minimal_code_policy.py)）。即使生成代码运行期报错或提前结束，评测器也会安全接管并正常完成回合。

---

## 3. 在线交互环境（供 RL / 迭代式 RSI 使用）

需要在线环境步进或采样 `dev` 回合时，请使用 [harness/dev_env.py](../harness/dev_env.py) 中的 `DevInteractionEnv`（参考模板 [examples/minimal_dev_interaction.py](../examples/minimal_dev_interaction.py)）：
* 仅暴露 `dev` 切分的 25 个官方初始状态/任务，物理隔离 `sealed_id` 与 `sealed_ood`；
* 每次 `reset()` 自动计入 `OnlineBudgetGuard`，超出 2,000 回合总预算上限会抛出 `BudgetExceededError`；
* 若 RL 奖励或 RSI 归因在 `dev` 训练期需要遥测信息，可传入 `allow_dev_privileged_info=True`，信息仅通过 `info["dev_privileged"]` 返回（严禁混入测试期 `obs_dict`）。

---

## 4. 最终出分与全组汇总

1. 每位组员在 `dev` 上确定最终权重、提示词或演化产物后，在**各自机器**上跑一次 `test_id`（100 题）和 `test_ood`（112 题）：
   ```bash
   MUJOCO_GL=egl python3 harness/evaluate.py --split test_id  --env-backend libero --policy-url http://127.0.0.1:8001 --demo-count 200 --gpu-hours 4.5 --out runs/my_method_id
   MUJOCO_GL=egl python3 harness/evaluate.py --split test_ood --env-backend libero --policy-url http://127.0.0.1:8001 --demo-count 200 --gpu-hours 4.5 --out runs/my_method_ood
   ```
   *(若用 `--policy-class`，直接把 `--policy-url ...` 换成 `--policy-class ...` 即可。)*
2. 把大家跑出来的 `runs/*_id` / `runs/*_ood` 结果目录收集到同一台机器上，运行聚合脚本生成最终 Markdown 对比表（含 Wilson 95% 区间、McNemar 检验、4 类失败归因）及 3 张矢量图：
   ```bash
   python3 harness/stats_and_report.py --runs runs/*_id  --out runs/final_report_id
   python3 harness/stats_and_report.py --runs runs/*_ood --out runs/final_report_ood
   ```
