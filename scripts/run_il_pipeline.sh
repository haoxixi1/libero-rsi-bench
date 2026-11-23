#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Imitation Learning (IL) 一键全流程实验流水线 (数据提取 + 训练 + Dev/Test_ID/Test_OOD 评测 + 报告)
#
# 模式 (--mode):
#   full       : 跑完整实验 (50条/任务 Full-Data IL + 20条/任务 Low-Data IL 消融 + 评测 + 对比图)
#   50only     : 仅跑 50条/任务 标准 IL 主实验 + 评测
#   smoke-mock : 零依赖快速自检 (在 Mock 仿真器上合成数据 -> 训 3 个 Epoch -> 跑完评测与出图)
#
# 配合防断连任务管理器启动 (推荐):
#   bash scripts/job_manager.sh run "il_exp" "bash scripts/run_il_pipeline.sh --mode full" 3 false
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="full"
ENV_BACKEND="libero"
EPOCHS=100
BATCH_SIZE=64
CONTROL_MODE="osc_delta_eef"
SAVE_VIDEOS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --env-backend)
      ENV_BACKEND="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --control-mode)
      CONTROL_MODE="$2"
      shift 2
      ;;
    --save-videos)
      SAVE_VIDEOS="true"
      shift 1
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${MUJOCO_GL:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    export MUJOCO_GL="egl"
  else
    export MUJOCO_GL="osmesa"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  fi
fi
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$MODE" == "smoke-mock" ]]; then
  ENV_BACKEND="mock"
  EPOCHS=3
  BATCH_SIZE=16
  DATA_DIR="$REPO_ROOT/datasets/il_mock_smoke"
  echo "=================================================================="
  echo "[SMOKE-MOCK 1/4] 生成 4 任务合成轨迹数据..."
  echo "=================================================================="
  "$PYTHON_BIN" baselines/prepare_dataset.py --source mock --max-demos-per-task 4 --out-dir "$DATA_DIR"

  echo "=================================================================="
  echo "[SMOKE-MOCK 2/4] 训练 Dual-Camera ACT 模型 (3 epochs,断点续训验证)..."
  echo "=================================================================="
  CKPT_DIR="$REPO_ROOT/checkpoints/il_mock_smoke"
  "$PYTHON_BIN" baselines/train_il.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$CKPT_DIR" \
    --policy-id "il_act_smoke" \
    --control-mode "$CONTROL_MODE" \
    --backbone cnn \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --resume

  echo "=================================================================="
  echo "[SMOKE-MOCK 3/4] 在 dev / test_id / test_ood 上验证评测流水线..."
  echo "=================================================================="
  KW_JSON="{\"checkpoint_path\": \"$CKPT_DIR/best.pt\"}"
  "$PYTHON_BIN" harness/evaluate.py --split dev      --limit 4 --env-backend mock --policy-class baselines.il_policy:ILChunkingPolicy --policy-kwargs "$KW_JSON" --demo-count 16 --gpu-hours 0.01 --out runs/il_smoke_dev
  "$PYTHON_BIN" harness/evaluate.py --split test_id  --limit 4 --env-backend mock --policy-class baselines.il_policy:ILChunkingPolicy --policy-kwargs "$KW_JSON" --demo-count 16 --gpu-hours 0.01 --out runs/il_smoke_id
  "$PYTHON_BIN" harness/evaluate.py --split test_ood --limit 4 --env-backend mock --policy-class baselines.il_policy:ILChunkingPolicy --policy-kwargs "$KW_JSON" --demo-count 16 --gpu-hours 0.01 --out runs/il_smoke_ood

  echo "=================================================================="
  echo "[SMOKE-MOCK 4/4] 生成统计对比报告与 SVG 图表..."
  echo "=================================================================="
  "$PYTHON_BIN" harness/stats_and_report.py --runs runs/il_smoke_id runs/il_smoke_ood --out runs/report_il_smoke
  echo "[OK] Smoke-Mock 全链路验证通过！报告位于: runs/report_il_smoke/REPORT_SUMMARY.md"
  exit 0
fi

# ==============================================================================
# 正式实验流程 (AutoDL RTX 5090 + LIBERO-plus)
# ==============================================================================
DATA_DIR="$REPO_ROOT/datasets/il_4tasks_50"
if [[ ! -f "$DATA_DIR/dataset_meta.json" ]]; then
  echo "=================================================================="
  echo "[STEP 1] 从 HuggingFaceVLA/libero 下载并提取 4 个核心任务的专家轨迹 (<=50条/任务)..."
  echo "=================================================================="
  "$PYTHON_BIN" baselines/prepare_dataset.py \
    --source hf \
    --repo-id HuggingFaceVLA/libero \
    --max-demos-per-task 50 \
    --out-dir "$DATA_DIR"
else
  echo "[STEP 1] 已检测到预处理数据集 $DATA_DIR/dataset_meta.json，跳过下载。"
fi

run_single_il_variant() {
  local tag="$1"           # e.g. il_act_50 or il_act_20
  local demos_per_task="$2" # 50 or 20
  local ckpt_dir="$REPO_ROOT/checkpoints/${tag}"

  if [[ ! -f "$ckpt_dir/ledger_meta.json" || ! -f "$ckpt_dir/latest.pt" ]]; then
    echo "=================================================================="
    echo "[TRAIN] 开始训练 ${tag} (${demos_per_task} demos/task, max ${EPOCHS} epochs, --resume)..."
    echo "=================================================================="
    "$PYTHON_BIN" baselines/train_il.py \
      --data-dir "$DATA_DIR" \
      --save-dir "$ckpt_dir" \
      --policy-id "$tag" \
      --control-mode "$CONTROL_MODE" \
      --max-demos-per-task "$demos_per_task" \
      --backbone resnet18 \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --resume
  else
    echo "[TRAIN] 已检测到 $ckpt_dir/latest.pt 与 ledger_meta.json，跳过 ${tag} 训练。"
  fi

  # 从 ledger_meta.json 读取真实 demo_count 与 gpu_hours
  local demo_cnt gpu_hrs
  demo_cnt=$("$PYTHON_BIN" -c "import json; d=json.load(open('$ckpt_dir/ledger_meta.json')); print(d['demo_count'])")
  gpu_hrs=$("$PYTHON_BIN" -c "import json; d=json.load(open('$ckpt_dir/ledger_meta.json')); print(d['gpu_hours'])")
  # 使用完整收敛的 100-epoch 余弦退火权重 latest.pt（避免小验证集 open-loop val_loss 在早期 epoch 4 过早停滞）
  local eval_ckpt="$ckpt_dir/latest.pt"
  if [[ ! -f "$eval_ckpt" ]]; then
    eval_ckpt="$ckpt_dir/best.pt"
  fi
  local kw_json="{\"checkpoint_path\": \"$eval_ckpt\"}"
  local vid_flag=()
  if [[ "$SAVE_VIDEOS" == "true" ]]; then
    vid_flag=(--save-videos)
  fi

  if [[ ! -f "runs/${tag}_dev/summary.json" ]]; then
    echo "=================================================================="
    echo "[EVAL 1/3] 评测 ${tag} on DEV split (100 episodes)..."
    echo "=================================================================="
    "$PYTHON_BIN" harness/evaluate.py \
      --split dev \
      --env-backend "$ENV_BACKEND" \
      --policy-class baselines.il_policy:ILChunkingPolicy \
      --policy-kwargs "$kw_json" \
      --demo-count "$demo_cnt" \
      --gpu-hours "$gpu_hrs" \
      --save-keyframes \
      --out "runs/${tag}_dev"
  else
    echo "[EVAL 1/3] 已检测到 runs/${tag}_dev/summary.json，跳过 DEV 评测。"
  fi

  if [[ ! -f "runs/${tag}_id/summary.json" ]]; then
    echo "=================================================================="
    echo "[EVAL 2/3] 评测 ${tag} on TEST_ID split (100 episodes, 域内测试集)..."
    echo "=================================================================="
    "$PYTHON_BIN" harness/evaluate.py \
      --split test_id \
      --confirm-unseal \
      --env-backend "$ENV_BACKEND" \
      --policy-class baselines.il_policy:ILChunkingPolicy \
      --policy-kwargs "$kw_json" \
      --demo-count "$demo_cnt" \
      --gpu-hours "$gpu_hrs" \
      --save-keyframes \
      "${vid_flag[@]}" \
      --out "runs/${tag}_id"
  else
    echo "[EVAL 2/3] 已检测到 runs/${tag}_id/summary.json，跳过 TEST_ID 评测。"
  fi

  if [[ ! -f "runs/${tag}_ood/summary.json" ]]; then
    echo "=================================================================="
    echo "[EVAL 3/3] 评测 ${tag} on TEST_OOD split (112 episodes, 4轴域外抗干扰测试集)..."
    echo "=================================================================="
    "$PYTHON_BIN" harness/evaluate.py \
      --split test_ood \
      --confirm-unseal \
      --env-backend "$ENV_BACKEND" \
      --policy-class baselines.il_policy:ILChunkingPolicy \
      --policy-kwargs "$kw_json" \
      --demo-count "$demo_cnt" \
      --gpu-hours "$gpu_hrs" \
      --save-keyframes \
      "${vid_flag[@]}" \
      --out "runs/${tag}_ood"
  else
    echo "[EVAL 3/3] 已检测到 runs/${tag}_ood/summary.json，跳过 TEST_OOD 评测。"
  fi
}

# 1. 运行标准 50 demos/task (共 200 条轨迹) 主实验
run_single_il_variant "il_act_50" 50

# 2. 若 mode == full，额外运行 20 demos/task (共 80 条轨迹) Low-Data 消融实验
if [[ "$MODE" == "full" ]]; then
  run_single_il_variant "il_act_20" 20
  echo "=================================================================="
  echo "[REPORT] 生成 IL (50 vs 20 demos) 汇总对比表与 3 张 SVG 矢量图..."
  echo "=================================================================="
  "$PYTHON_BIN" harness/stats_and_report.py --runs runs/il_act_50_id runs/il_act_20_id --out runs/report_il_id
  "$PYTHON_BIN" harness/stats_and_report.py --runs runs/il_act_50_ood runs/il_act_20_ood --out runs/report_il_ood
else
  "$PYTHON_BIN" harness/stats_and_report.py --runs runs/il_act_50_id runs/il_act_50_ood --out runs/report_il_50
fi

echo "=================================================================="
echo "[SHOWCASE] 导出 4 个基础任务 (ID & OOD) 的双视角仿真视频 (.mp4)、动图 (.gif) 与 6 帧时序长图 (.png)..."
echo "=================================================================="
"$PYTHON_BIN" scripts/export_visual_showcase.py \
  --checkpoint "$REPO_ROOT/checkpoints/il_act_50/latest.pt" \
  --split test_id \
  --episodes-per-task 1 \
  --env-backend "$ENV_BACKEND" \
  --out "$REPO_ROOT/runs/visual_showcase_id" || true

"$PYTHON_BIN" scripts/export_visual_showcase.py \
  --checkpoint "$REPO_ROOT/checkpoints/il_act_50/latest.pt" \
  --split test_ood \
  --episodes-per-task 1 \
  --env-backend "$ENV_BACKEND" \
  --out "$REPO_ROOT/runs/visual_showcase_ood" || true

echo "=================================================================="
echo "[ALL FINISHED] IL 实验全部跑完！"
echo "  -> 结果目录: runs/il_act_50_id, runs/il_act_50_ood, runs/visual_showcase_id, runs/visual_showcase_ood"
echo "  -> 在 Mac 本地运行以下命令即可一键拉回所有结果、视频与图表："
echo "     bash scripts/autodl_connect.sh pull"
echo "=================================================================="
