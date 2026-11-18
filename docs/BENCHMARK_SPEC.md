# `libero-rsi-bench` 统一仿真评测基准与实验规范（v1.1）

> 面向 **AIR5023 Course Project**：本仓库仅定义**权威仿真基准、任务切分锁、统一观测与步数契约、在线预算守卫以及统计出图口径**，不对各组员所选用的具体模型架构、训练库或策略代码作限制。任何 **模仿学习（IL）**、**强化学习（RL）**、**智能体策略（Agentic Policy）** 与 **自我改进（RSI）** 只要实现统一评测接口并遵守对应方法族的准入预算，即可在同一基准上公平对决。

---

## 1. 设计动机与继承原则

前期 `FetchPickAndPlace-v1` 实验方案确立了四项核心评测纪律，本基准将其完整继承并升级至更贴近真实桌面操作的 **`LIBERO + LIBERO-Plus`** 生态：

1. **用原生域外扰动解决“任务饱和”**：原版 `LIBERO` 作为域内基准（ID），`sylvestf/LIBERO-plus` 提供的七维物理与视觉扰动作为域外基准（OOD），无需混合异构的 `RoboTwin`（SAPIEN 双臂 14-DoF），全组统一使用 **MuJoCo 2.3.7 + robosuite 1.4.1（单臂 7-DoF Franka Panda，20Hz `OSC_POSE` 增量控制）**。
2. **三切分隔离与 SHA-256 密封防偷看**：`burned(8)`、`dev(100)` 与一次性密封考卷 `sealed_id(100)`、`sealed_ood(112)` 由 [protocol/generate_splits.py](../protocol/generate_splits.py) 锁定并计算 `sealed_sha256`；任何对 `sealed_*` 的读取必须显式授权并写入审计日志 `runs/sealed_reads.log`。
3. **基于有限样本功效分析设计判据**：显式量化 $n=100$ 下配对检验的检出率约束，区分“门控停摆”与“无门控崩塌”。
4. **成本一等指标**：统一账本 `CostLedger` 将示范数、在线回合/步数、算力耗时与 API 调用/Token 与成功率并列报告。

---

## 2. 核心任务集与域外扰动轴

由 [protocol/tasks_manifest.json](../protocol/tasks_manifest.json) 锁定的 **4 个代表性核心任务**（覆盖 LIBERO 四大官方套件）：

| 任务键名 | 所属套件 | `clean_base_name` | 标准语言指令 | 任务类型 | 原生步数 `1.0x` | 宽容窗口 `1.5x` | 宏决策上限 |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `spatial_t0` | `libero_spatial` | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | `pick up the black bowl between the plate and the ramekin and place it on the plate` | 桌面空间抓放 | **220** | **450** | 15 |
| `object_t0` | `libero_object` | `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | `pick up the alphabet soup and place it in the basket` | 跨篮沿抓放 | **280** | **450** | 15 |
| `goal_t0` | `libero_goal` | `open_the_middle_drawer_of_the_cabinet` | `open the middle drawer of the cabinet` | 铰接拉抽屉 | **300** | **450** | 15 |
| `long_t0` | `libero_10` | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | `put both the alphabet soup and the tomato sauce in the basket` | 双物体长程 | **520** | **750** | 25 |

### `LIBERO-Plus` 四条核心 OOD 扰动轴（Level 2–4）
`sealed_ood` 针对每个任务在以下 4 个维度各抽 7 个变体槽位（共 $4 \times 4 \times 7 = 112$ 回合），通过 `LIBERO-plus` 官方 `task_classification.json` 映射：
1. **`objects_layout`**（对应 `Objects Layout`）：目标位置偏移与干扰物注入。
2. **`robot_init_state`**（对应 `Robot Initial States`）：机械臂初始关节与末端位姿扰动。
3. **`language_instruction`**（对应 `Language Instructions`）：同义改写指令。
4. **`camera_visual`**（对应 `Camera Viewpoints` / `Light Conditions` / `Background Textures` / `Sensor Noise`）：相机视角、光照、纹理与图像噪声。

---

## 3. 数据切分与密封考卷协议

| 切分 | 回合数 | 种子与状态来源 | 允许用途 | 禁止用途 |
| :--- | ---: | :--- | :--- | :--- |
| `burned` | 8 | 4 任务 × 2 动态种子（`1000–1031`） | 接口调试、冒烟检查 | 严禁计入 Dev / 测试成绩 |
| `dev` | **100** | 4 任务 × 官方 `init_state_id 0..24`（`2000–2324`） | 验证早停、RL 在线探索、RSI 迭代归因与门控 | 严禁突破 2,000 回合预算上限 |
| `sealed_id` | **100** | 4 任务 × 官方 `init_state_id 25..49`（`4025–4349`） | **所有方法冻结后一次性开卷最终评测** | 严禁训练、调参、RSI 门控读取 |
| `sealed_ood` | **112** | 4 任务 × 4 OOD 轴 × 7 变体（`10000–13306`） | **所有方法冻结后一次性开卷最终评测** | 严禁训练、调参、RSI 门控读取 |

---

## 4. 统一观测契约与 `LIBERO-plus` 工程纠偏规则

所有策略通过 [harness/env_adapter.py](../harness/env_adapter.py) 访问仿真环境，统一执行以下不可绕过的物理与数据契约：

1. **严禁读取真值物理状态**：`_sanitize_obs()` 剥离全部 `object-state`、物体坐标真值、`task_type` 标签及 `sim.data`，杜绝执行期“开天眼”。同时向策略无差异暴露纯机器人本体感觉：**8 维末端状态 `state`**（`eef_pos + axisangle + gripper_qpos`）与 **7 维关节状态 `joint_pos` / `joint_vel`**（单位 `rad`、`rad/s`，零特权泄露）。若 RL/RSI 在 `dev` 训练集内部需要失败归因或稠密奖励，可通过 `DevInteractionEnv(..., allow_dev_privileged_info=True)` 从 `info["dev_privileged"]` 获取（物理隔离于测试期）。
2. **多控制模式统一按 50ms 物理步计量（不绑定方法流派）**：策略可通过 `control_mode` 声明（`GET /health` 自动协商）或 CLI `--control-mode` 自由选用 **`osc_delta_eef`**（7 维末端 `[-1, 1]` 增量，默认）、**`abs_eef`**（7 维世界系绝对末端位姿）、**`abs_joint_pos` / `delta_joint_pos`**（8 维 Panda 关节角控制）、**`macro_rel_eef`**（4/7 维物理米制宏相对位移）或 **`macro_target_eef`**（4/7 维世界系目标路点）。其中两个 `macro_*` 模式共用评测底座内锁死的闭环 P 伺服控制器，底层实际走过的每一个 20Hz 物理子步均等额计入 `steps_taken` 与 `SR@1.0x / 1.5x` 窗口。
3. **统一相机预处理与静止帧**：双视角图像（`agentview_image`、`robot0_eye_in_hand_image`）统一做 **180° 旋转**（`[::-1, ::-1]`）并规范为 `uint8`；每次 `reset()` 后执行 **10 步中性静止动作**。
4. **两条隔离感知轨**：
   - **`rgbd_projection`（默认主轨，纯视觉闭环）**：提供双视角 256×256 RGB、**经 `get_real_depth_map` 转换的真实物理米制 Depth（`float32`, meters）**、与 `[::-1, ::-1]` 图像对齐的标称相机内外参（`camera_intrinsics` / `camera_extrinsics`，在 `camera_visual` OOD 下不泄露扰动外参），以及上述末端与关节本体状态。任何需要 3D 坐标的方法必须自行从图像/深度反投影（`harness/math_kinematics.py` 提供 `deproject_pixel_to_world`）。
   - **`noisy_keypoints`（轻量消融轨）**：向有限感知接口注入按种子确定性生成的 $\sigma = 10\text{ mm}$ 高斯噪声（`camera_visual` OOD 下放大至 $2.2\sigma$），用于低算力复现与噪声鲁棒性消融。
5. **`LIBERO-plus` 原生陷阱修复（已实测验证）**：
   - **干净任务独立加载**：`LIBERO-plus` 的任务注册表仅包含 10,030 个扰动变体，`burned` / `dev` / `sealed_id` 须直接加载 `<clean_base_name>.bddl` 与 `<clean_base_name>.pruned_init`。
   - **指令后缀污染清洗**：非 `language_instruction` 扰动任务在 `LIBERO-plus` 中的 `task.language` 会拼接 `_view_..._noise_...` 等文件名后缀；除 `language_instruction` 外，其余切分与扰动轴统一强制注入标准指令。
   - **保护 `Robot Initial States` 初始关节**：`set_init_state(inits[0])` 会把 `reset()` 生成的扰动机械臂姿态覆盖回默认姿；`LiberoPlusEnvAdapter` 在 `robot_init_state` 下先保留 `sim.data.qpos[:9]` 再恢复并刷新观测。

---

## 5. 双时间窗步数协议（解决 Agentic 步数需求高的问题）

由于调用宏技能/重规划的 Agentic 策略与代码级 RSI 在 20Hz 控制下需要分段对齐与闭环重抓，其物理步数天然比端到端连续策略多约 **30%–50%**。为避免偏袒任何一方，基准采用 **100% 沿用官方成功判定 + 统一步数上限 + 单次运行导出双时间窗**：

1. **统一运行上限（`1.5x`）**：所有方法单回合统一运行至 `extended_steps`（单阶段 **450 步** / 长程 **750 步**）。
2. **完全原版 LIBERO 成功判据（`hold_success_steps = 1`）**：沿用原版 `done or env.check_success()`，一旦满足即终止回合，并记录达成成功的精确步数 $t_{\text{succ}}$。
3. **同时报告双指标**：
   - **`SR@1.0x`（原版官方同口径）**：$t_{\text{succ}} \le 220 / 280 / 300 / 520$ 的成功率——与开源社区标准 LIBERO/LIBERO-Plus 评测口径完全一致；
   - **`SR@1.5x`（宽容完成口径）**：$t_{\text{succ}} \le 450$（长程 $\le 750$）的成功率——衡量允许闭环纠错与重规划后的真实完成率。
4. **宏观决策轮数硬上限**：`agentic` 策略由 `rollout_episode` 强制校验 `macro_turns < max_macro_turns`（`15` / `25` 轮），超出后只能输出静止占位动作直至回合结束。

---

## 6. 四大家族统一准入与预算规则（方法无关）

各组员可自由选择任意开源框架、模型结构或提示词工程，但必须严格遵守所属方法族的**资源与交互预算上限**：

| 方法族 | 离线专家示范准入 | 在线 `dev` 训练/迭代交互上限 | 测试期交互与汇报约束 |
| :--- | :--- | :--- | :--- |
| **模仿学习（IL）** | 仅可使用 4 个任务共享的 `50 demos/task`（可选增加 `20 demos` 少样本组） | **训练过程 `0` 在线交互**（仅允许周期性在 `dev` 跑验证选早停权重） | 不限网络架构；须上报离线 Demo 数与训练 GPU 小时 |
| **强化学习（RL）** | 可选使用与 IL 同一批离线示范做热启动或离线 Buffer | **严禁超过 `2,000` 个在线 `dev` 回合**（建议通过 `DevInteractionEnv` 硬锁每任务 `500` 回合） | 不限 RL 算法（PPO/SAC/RLPD/VLA-RL 等）；须上报总在线步数与 GPU 小时 |
| **自我改进（RSI）** | 可选零样本或读相同示范描述 | **迭代归因与门控评测合计严禁超过 `2,000` 个在线 `dev` 回合**（例如 `20` 轮 × `dev` 100 题） | 不限改代码、改原语或自采微调数据；须上报迭代轨迹与全部废弃提议的 Token 成本 |
| **Agentic Policy** | 零训练（Zero-shot 或提示词内至多 2 条文本示例） | **训练期 `0` 在线交互**（仅允许用 `burned` 8 题调通接口） | 单回合宏决策不超 `15 / 25` 轮；**若使用自定义技能原语，必须额外提交 `--no-llm` 纯规则兜底对照组**以剥离原语自身贡献 |

---

## 7. $n=100$ 统计功效与五层汇总口径

在 `dev`（$n=100$）或 `sealed_id`（$n=100$）上使用精确 McNemar 检验（假设基础分歧率 `discordant_base = 0.06`），由 [harness/stats_and_report.py](../harness/stats_and_report.py) 计算的有限样本检出率（Power）为：

| 真实相对提升 $\Delta\text{SR}$ | `+1 pt` | `+5 pt` | `+10 pt` | `+15 pt` | `+20 pt` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 双侧 $\alpha = 0.05$ | 2.5% | 16.3% | 49.7% | 79.9% | 94.6% |
| 单侧 $\alpha = 0.10$ | 9.9% | 37.8% | 75.5% | 93.9% | 99.0% |

所有参赛策略最终由 `stats_and_report.py` 统一输出：
1. `SR@1.0x` / `SR@1.5x` 及 **95% Wilson 置信区间**；
2. `Dev → Sealed_ID` 过拟合裂口与 `Sealed_ID → Sealed_OOD` 泛化裂口；
3. 共享种子上的**精确双侧 McNemar 配对检验**；
4. 四类自动失败簇分布（`approach_miss` / `grasp_slip` / `place_or_articulate_fail` / `timeout_incomplete`）；
5. 生命周期四维成本账本及 3 张矢量图（`horizon_curve.svg` / `failure_clusters.svg` / `cost_pareto.svg`）。
