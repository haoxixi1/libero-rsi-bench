#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 防断连训练与评测任务管理工具 (Job Manager)
#
# 核心机制:
#   1. Tmux 持久化会话 + SIGINT/SIGTERM 信号捕获保存断点 (SSH 断开绝不中断)
#   2. 崩溃自动恢复守护循环 (Auto-Resume Watchdog): 若遇 OOM/偶发抖动退出，自动从 latest.pt 续训
#   3. 统一日志流 + JSON 状态心跳文件 + GPU 实时监控
#   4. 支持实验完成后自动关机 (节省 AutoDL 5090 计费: --auto-shutdown)
#
# 常用命令:
#   bash scripts/job_manager.sh run "<name>" "<command>" [--auto-Restart] [--auto-shutdown]
#   bash scripts/job_manager.sh status
#   bash scripts/job_manager.sh logs [name]
#   bash scripts/job_manager.sh attach [name]
#   bash scripts/job_manager.sh stop <name>
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
STATE_DIR="$LOG_DIR/job_state"
ENV_NAME="${ENV_NAME:-libero-bench}"

mkdir -p "$LOG_DIR" "$STATE_DIR"

ACTION="${1:-status}"

print_status() {
  echo "=========================================================================="
  echo " [1] GPU 状态 (nvidia-smi)"
  echo "=========================================================================="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
  else
    echo "  (当前机器未检测到 nvidia-smi)"
  fi

  echo ""
  echo "=========================================================================="
  echo " [2] 活跃的 Tmux 训练/评测会话"
  echo "=========================================================================="
  if command -v tmux >/dev/null 2>&1; then
    tmux ls 2>/dev/null || echo "  (当前无运行中的 tmux 会话)"
  else
    echo "  (未安装 tmux)"
  fi

  echo ""
  echo "=========================================================================="
  echo " [3] 任务状态账本 ($STATE_DIR)"
  echo "=========================================================================="
  shopt -s nullglob
  local files=("$STATE_DIR"/*.status)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "  (暂无任务记录)"
  else
    for f in "${files[@]}"; do
      echo "  -> $(basename "$f" .status): $(cat "$f")"
    done
  fi

  echo ""
  echo "=========================================================================="
  echo " [4] 断点权重文件 (checkpoints/)"
  echo "=========================================================================="
  if [[ -d "$REPO_ROOT/checkpoints" ]]; then
    find "$REPO_ROOT/checkpoints" -maxdepth 3 -name "*.pt" -printf "  %TY-%Tm-%Td %TH:%TM  %p (%k KB)\n" 2>/dev/null | sort || \
      find "$REPO_ROOT/checkpoints" -maxdepth 3 -name "*.pt" 2>/dev/null
  else
    echo "  (暂无 checkpoints 目录)"
  fi
  echo "=========================================================================="
}

launch_job() {
  local job_name="$1"
  local cmd_str="$2"
  local max_retries="${3:-3}"
  local auto_shutdown="${4:-false}"

  local session_name="job_${job_name}"
  local log_file="$LOG_DIR/${job_name}.log"
  local status_file="$STATE_DIR/${job_name}.status"
  local runner_script="$STATE_DIR/run_${job_name}.sh"

  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$session_name" 2>/dev/null; then
    echo "[WARN] 会话 '$session_name' 已在运行中！请先用 'bash scripts/job_manager.sh status' 查看，或 'stop $job_name' 停止。"
    exit 1
  fi

  cat > "$runner_script" <<EOF
#!/usr/bin/env bash
set -uo pipefail

cd "$REPO_ROOT"
export HF_HOME="/root/autodl-tmp/hf_cache"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export NO_PROXY="localhost,127.0.0.1,mirrors.aliyun.com,pypi.tuna.tsinghua.edu.cn,mirrors.cloud.tencent.com,hf-mirror.com"
export no_proxy="\$NO_PROXY"
export MUJOCO_GL="egl"
export PYTHONUNBUFFERED=1
[ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1 || true

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/root/miniconda3/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME" 2>/dev/null || true
fi

attempt=1
max_retries=$max_retries

while [[ \$attempt -le \$max_retries ]]; do
  start_ts=\$(date '+%Y-%m-%d %H:%M:%S')
  echo "RUNNING (attempt \$attempt/\$max_retries, started at \$start_ts)" > "$status_file"
  echo "==================================================================" | tee -a "$log_file"
  echo "[\$start_ts] STARTING JOB '$job_name' (Attempt \$attempt/\$max_retries)" | tee -a "$log_file"
  echo "COMMAND: $cmd_str" | tee -a "$log_file"
  echo "==================================================================" | tee -a "$log_file"

  bash -c "$cmd_str" 2>&1 | tee -a "$log_file"
  exit_code=\${PIPESTATUS[0]}
  end_ts=\$(date '+%Y-%m-%d %H:%M:%S')

  if [[ \$exit_code -eq 0 ]]; then
    echo "COMPLETED (finished at \$end_ts)" > "$status_file"
    echo "[\$end_ts] JOB '$job_name' SUCCEEDED (exit 0)." | tee -a "$log_file"
    if [[ "$auto_shutdown" == "true" ]]; then
      echo "[\$end_ts] Auto-shutdown enabled. Syncing filesystem and shutting down in 60s..." | tee -a "$log_file"
      sync
      sleep 60
      shutdown -h now
    fi
    exit 0
  fi

  echo "[\$end_ts] [WARN] JOB '$job_name' exited with code \$exit_code." | tee -a "$log_file"
  attempt=\$((attempt + 1))
  if [[ \$attempt -le \$max_retries ]]; then
    echo "[\$end_ts] Waiting 10s before auto-resuming from latest checkpoint..." | tee -a "$log_file"
    sleep 10
  fi
done

fail_ts=\$(date '+%Y-%m-%d %H:%M:%S')
echo "FAILED (exit_code=\$exit_code at \$fail_ts after \$max_retries attempts)" > "$status_file"
echo "[\$fail_ts] [ERROR] JOB '$job_name' FAILED after \$max_retries attempts." | tee -a "$log_file"
exit 1
EOF
  chmod +x "$runner_script"

  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "$session_name" "bash $runner_script"
    echo "[OK] 任务 '$job_name' 已在后台 Tmux 会话 '$session_name' 启动！"
    echo "  -> 实时查看日志: bash scripts/job_manager.sh logs $job_name"
    echo "  -> 进入交互终端: bash scripts/job_manager.sh attach $job_name  (退回后台按 Ctrl+B 再按 D)"
    echo "  -> 查看运行状态: bash scripts/job_manager.sh status"
  else
    nohup bash "$runner_script" >/dev/null 2>&1 &
    echo "[OK] 任务 '$job_name' 已通过 nohup 后台启动 (PID=$!)，日志: $log_file"
  fi
}

case "$ACTION" in
  status)
    print_status
    ;;
  run)
    JOB_NAME="${2:-}"
    JOB_CMD="${3:-}"
    MAX_RETRIES="${4:-3}"
    AUTO_SHUTDOWN="${5:-false}"
    if [[ -z "$JOB_NAME" || -z "$JOB_CMD" ]]; then
      echo "用法: bash scripts/job_manager.sh run <任务名> \"<执行命令>\" [最大重试次数=3] [完成后自动关机=false|true]"
      exit 1
    fi
    launch_job "$JOB_NAME" "$JOB_CMD" "$MAX_RETRIES" "$AUTO_SHUTDOWN"
    ;;
  logs)
    JOB_NAME="${2:-}"
    if [[ -z "$JOB_NAME" ]]; then
      echo "可用日志文件:"
      ls -lh "$LOG_DIR"/*.log 2>/dev/null || echo "  (暂无日志)"
      exit 0
    fi
    tail -n 100 -f "$LOG_DIR/${JOB_NAME}.log"
    ;;
  attach)
    JOB_NAME="${2:-}"
    if [[ -z "$JOB_NAME" ]]; then
      echo "用法: bash scripts/job_manager.sh attach <任务名>"
      exit 1
    fi
    exec tmux attach -t "job_${JOB_NAME}"
    ;;
  start-all)
    AUTO_SHUTDOWN="${2:-false}"
    launch_job "il_exp" "bash scripts/run_il_pipeline.sh --mode full" 3 "$AUTO_SHUTDOWN"
    ;;
  start-50)
    AUTO_SHUTDOWN="${2:-false}"
    launch_job "il_exp_50" "bash scripts/run_il_pipeline.sh --mode 50only" 3 "$AUTO_SHUTDOWN"
    ;;
  start-smoke)
    launch_job "il_smoke" "bash scripts/run_il_pipeline.sh --mode smoke-mock" 1 false
    ;;
  stop)
    JOB_NAME="${2:-}"
    if [[ -z "$JOB_NAME" ]]; then
      echo "用法: bash scripts/job_manager.sh stop <任务名>"
      exit 1
    fi
    session_name="job_${JOB_NAME}"
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$session_name" 2>/dev/null; then
      # 发送 Ctrl-C 让 Python 捕获 SIGINT 保存 checkpoint 后优雅退出
      tmux send-keys -t "$session_name" C-c
      sleep 3
      tmux kill-session -t "$session_name" 2>/dev/null || true
      echo "STOPPED_BY_USER ($(date '+%Y-%m-%d %H:%M:%S'))" > "$STATE_DIR/${JOB_NAME}.status"
      echo "[OK] 已优雅停止任务 '$JOB_NAME'"
    else
      echo "[WARN] 未找到运行中的会话 '$session_name'"
    fi
    ;;
  *)
    echo "用法:"
    echo "  bash scripts/job_manager.sh status                                        # 查看 GPU、Tmux 会话、任务进度与权重"
    echo "  bash scripts/job_manager.sh run <name> \"<cmd>\" [retries=3] [shutdown=false] # 启动防断连自动恢复后台任务"
    echo "  bash scripts/job_manager.sh logs <name>                                   # 实时追踪任务日志 (Ctrl+C 退出查看不影响任务)"
    echo "  bash scripts/job_manager.sh attach <name>                                 # 挂接到 Tmux 会话 (Ctrl+B D 退回后台)"
    echo "  bash scripts/job_manager.sh stop <name>                                   # 发送 SIGINT 保存断点并停止任务"
    ;;
esac
