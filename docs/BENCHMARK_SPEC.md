# `libero-rsi-bench` 小组实验基准与评测协议

本基准用于在同一个仿真环境下（**`LIBERO` + `LIBERO-Plus`**，7 自由度 Franka Panda 机械臂）公平对比 **模仿学习（IL）、强化学习（RL）、Agentic Policy、自我改进（RSI）** 四类方法。不限制组员具体选什么模型架构或训练库，只需遵守下面的数据范围、预算上限与统一评测协议即可。

---

## 1. 评测任务与抗干扰（OOD）设置

共精选 **4 个代表性任务**（4 个任务视频见 [docs/videos/](videos/)）：

| 任务 ID | 任务说明 | 英文指令（`instruction`） | 标准步数 `1.0x` | 放宽上限 `1.5x` *(暂定)* | 任务演示 |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `spatial_t0` | 把盘子与小碗之间的黑碗放到盘子上 | `pick up the black bowl between the plate and the ramekin and place it on the plate` | **220** | **450** | [spatial_t0.mp4](videos/spatial_t0.mp4) |
| `object_t0` | 抓起字母汤罐头放进篮子里 | `pick up the alphabet soup and place it in the basket` | **280** | **450** | [object_t0.mp4](videos/object_t0.mp4) |
| `goal_t0` | 拉开木柜子的中间抽屉 | `open the middle drawer of the cabinet` | **300** | **450** | [goal_t0.mp4](videos/goal_t0.mp4) |
| `long_t0` | 先后把字母汤和番茄酱两罐放进篮子 | `put both the alphabet soup and the tomato sauce in the basket` | **520** | **750** | [long_t0.mp4](videos/long_t0.mp4) |

> **注**：表中的 **放宽上限 `1.5x`（`450 / 750` 步）目前为暂定值**，后续可根据各组员方法在 `dev` 上的实际步数需求在 [protocol/tasks_manifest.json](../protocol/tasks_manifest.json) 的 `extended_steps` 字段中统一调整。

除标准域内测试（**ID**）外，最终还会从 `LIBERO-Plus` 抽取 **4 类抗干扰域外测试（OOD，共 112 题）**：
1. **`objects_layout`**：桌上物体位置偏移或注入额外干扰物；
2. **`robot_init_state`**：机械臂初始关节位姿偏移；
3. **`language_instruction`**：语言指令同义改写；
4. **`camera_visual`**：相机视角偏转、光照变化、桌面材质与画面噪声。

---

## 2. 离线训练数据下载与使用范围

> **红线**：离线训练或热启动**只能使用上述 4 个任务的专家示范（每任务最多 `50` 条轨迹，合计最多 `200` 条）**。严禁使用其它 LIBERO 任务的轨迹，也严禁直接加载在 LIBERO 上微调过的权重（如 `*-finetuned-libero-*`）；通用预训练视觉/VLM 基座（如 CLIP、YOLO、SAM、通用基础 VLA/VLM）可以使用，并在汇报中写明即可。

### 统一数据集下载与筛选（LeRobot `no_noops` 去停顿清洗版）
全组统一使用清洗后的 `HuggingFaceVLA/libero` 数据集（去掉了遥操作停顿帧）：
```bash
huggingface-cli download HuggingFaceVLA/libero --repo-type dataset --local-dir ./datasets/libero_lerobot
```
按任务筛选对应 4 个子集的指定 `task_index`（每任务各取 `<=50` 条）：
* `libero_spatial` $\rightarrow$ `task_index == 0`
* `libero_object` $\rightarrow$ `task_index == 0`
* `libero_goal` $\rightarrow$ `task_index == 0`
* `libero_10` $\rightarrow$ `task_index == 5`

> **不同动作空间的训练标签提取提示**：
> - 用 **`osc_delta_eef`**（7 维相对增量）：直接拟合数据集 `action` 列；
> - 用 **`abs_eef`**（7 维绝对末端位姿）：用第 $t+1$ 帧 `observation.states.ee_state`（6 维）+ 第 $t$ 帧 `action[6]`（夹爪）；
> - 用 **`abs_joint_pos`**（8 维绝对关节角）：用第 $t+1$ 帧 `observation.states.joint_state`（7 维）+ 第 $t$ 帧 `action[6]`（夹爪）。

---

## 3. 四类方法的训练 / 进化预算规定

**单任务分开训（4 个模型各自对应一个任务）或多任务合训（1 个模型通吃 4 个任务）都可以**，最终实验表格里备注清楚模型数量即可。四类方法按各自特性设置上限：

| 方法类别 | 离线专家数据 | 在线交互 / 训练 / 迭代上限 | 说明 |
| :--- | :--- | :--- | :--- |
| **模仿学习（IL）** | 每任务 ≤ `50` 条轨迹（共计 ≤ `200` 条） | **最多训练 `100 epochs`**（或等效 `50k` gradient steps）；训练期 `0` 在线仿真步 | 可定期在 `dev` 上测一次挑最佳权重 |
| **强化学习（RL）** | 可不用，或仅用同一批 `≤50` 条/任务做热启动 | 仅允许使用 `dev` 集在线交互：**每任务最多 `500` 回合 / 约 20 万物理步（4 任务合计最多 `2,000` 回合）** | 请使用 [harness/dev_env.py](../harness/dev_env.py) 的 `DevInteractionEnv` 自动计次 |
| **Agentic Policy**<br>*(无进化基线)* | `0` 训练（最多放 `2` 条文本示例在 Prompt 中） | 训练期 `0` 在线试错（仅用 `dev` 少量题调通接口）；单回合高层决策上限 `15` 次（`long_t0` `25` 次） | 用于对照证明后续自我进化（RSI）的增益 |
| **RSI**<br>*(Agent 自我进化)* | 同 Agentic（禁止偷看测试集） | **必须与 Agentic 使用同款底座大模型**；最多允许迭代 **`10` 轮**，迭代期 `dev` 在线试错与评测合计 **≤ `2,000` 回合** | 允许沉淀动作技能、接入 YOLO/SAM 视觉工具、构建反思记忆或自改规划代码 |

---

## 4. 评测协议与最终对比指标

### 数据切分怎么用（组员各自本机跑分）
* **`dev`（100 题，每任务 25 个初始状态）**：平时写完代码可加 `--limit 4` 快速冒烟，正式开发阶段用于 IL 选最佳 Epoch、RL 在线训练、RSI 自我改进与查看失败案例（注意在线交互总量勿超 2,000 回合）。
* **`test_id`（100 题）+ `test_ood`（112 题）**：最终测试集（平时调参和迭代**严禁使用或偷看**）。每位组员方法定稿后，直接在各自机器上分别运行 `--split test_id` 和 `--split test_ood` 跑出结果即可。

### 为什么设 `1.0x` 与 `1.5x` 两个步数上限？
由于 Agentic / RSI 每次决策驱动机械臂分段对齐或重抓，物理步数通常比 20Hz 神经网络模型多 30%–50%。为保证公平：
1. **统一运行规则**：评测时每个回合统一允许最多跑到 `1.5x` 放宽上限（目前暂定单阶段 `450` 步、`long_t0` `750` 步，后续可按全组实测情况调整 [tasks_manifest.json](../protocol/tasks_manifest.json) 的 `extended_steps`），只要达成任务判定立即提前结束并记下精确步数 $t_{\text{succ}}$。
2. **一趟跑出双成绩**：
   * **`SR@1.0x`**：$t_{\text{succ}} \le 220 / 280 / 300 / 520$ 的成功率（对齐原版 LIBERO 官方步数要求）；
   * **`SR@1.5x`**：$t_{\text{succ}} \le 450 / 750$（暂定）的成功率（放宽步数后真正把任务做完的概率）。

### 接口与最终汇总
* 具体观测字典（含图像、米制深度、相机内外参、末端与 `joint_pos` 本体状态）、**6 种可选 `control_mode`**（相对/绝对 EEF、相对/绝对关节角、Agent 宏相对位移/绝对路点）及 HTTP 解耦启动命令，见 [组员接入规范](INTEGRATION_GUIDE.md)。
* 运行 [harness/evaluate.py](../harness/evaluate.py) 时加上 `--save-videos` 或 `--save-keyframes` 可直接导出双视角 MP4 回放或关键帧 PNG（方便做演示视频与 RSI 视觉反思）。
* 全部跑完后用 [harness/stats_and_report.py](../harness/stats_and_report.py) 汇总，最终横向对比 4 项结果：
  1. **成功率与抗干扰掉点**：`ID` 与 `OOD` 下的 `SR@1.0x` / `SR@1.5x`；
  2. **执行效率**：平均每题消耗的物理步数；
  3. **失败原因分布**：自动统计 4 类典型失败（`approach_miss` 没够着、`grasp_slip` 抓起又滑落、`place_or_articulate_fail` 放歪/没拉开、`timeout_incomplete` 做一半超时）；
  4. **开销账本**：离线示范数、训练/进化 GPU 小时数、在线回合数以及大模型 API Token 消耗。
