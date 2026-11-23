#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# AutoDL SSH 免密登录配置与远程部署工具 (Mac 本地端执行)
# 用法:
#   1. 配置免密 + 同步代码:
#      bash scripts/autodl_connect.sh setup "ssh -p 38123 root@connect.bjb1.seetacloud.com"
#   2. 登录远程终端:
#      bash scripts/autodl_connect.sh shell
#   3. 同步本地修改到远程:
#      bash scripts/autodl_connect.sh push
#   4. 拉取远程 runs/ 和 checkpoints/ 结果到本地:
#      bash scripts/autodl_connect.sh pull
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY_DIR="${SSH_KEY_DIR:-$HOME/.ssh}"
SSH_CFG="$SSH_KEY_DIR/config"
SSH_ALIAS="autodl-libero"
REMOTE_DIR="/root/autodl-tmp/libero-rsi-bench"

ACTION="${1:-help}"
SSH_CMD_RAW="${2:-}"

push_code() {
  echo "[SYNC] 正在将本地仓库同步至 $SSH_ALIAS:$REMOTE_DIR ..."
  ssh -F "$SSH_CFG" "$SSH_ALIAS" "mkdir -p $REMOTE_DIR"
  rsync -avz --progress -e "ssh -F $SSH_CFG" \
    --exclude '.git/' \
    --exclude '.venv*/' \
    --exclude '.libero_config/' \
    --exclude 'third_party/' \
    --exclude 'datasets/' \
    --exclude 'checkpoints/' \
    --exclude 'runs/' \
    --exclude 'logs/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "$REPO_ROOT/" "$SSH_ALIAS:$REMOTE_DIR/"
  echo "[OK] 代码已同步到 $REMOTE_DIR"
}

pull_results() {
  echo "[PULL] 正在从 $SSH_ALIAS:$REMOTE_DIR 拉取评测结果 (runs/) 与日志 (logs/) ..."
  mkdir -p "$REPO_ROOT/runs" "$REPO_ROOT/logs"
  rsync -avz --progress -e "ssh -F $SSH_CFG" "$SSH_ALIAS:$REMOTE_DIR/runs/" "$REPO_ROOT/runs/" || true
  rsync -avz --progress -e "ssh -F $SSH_CFG" "$SSH_ALIAS:$REMOTE_DIR/logs/" "$REPO_ROOT/logs/" || true
  echo "[OK] 结果已拉取到本地 $REPO_ROOT/runs"
}

case "$ACTION" in
  push)
    push_code
    ;;
  pull)
    pull_results
    ;;
  shell)
    exec ssh -F "$SSH_CFG" "$SSH_ALIAS"
    ;;
  cmd)
    shift
    exec ssh -F "$SSH_CFG" "$SSH_ALIAS" "$@"
    ;;
  *)
    echo "用法:"
    echo "  bash scripts/autodl_connect.sh push          # 同步本地代码到 AutoDL"
    echo "  bash scripts/autodl_connect.sh pull          # 拉取 AutoDL 的 runs/ 和 logs/"
    echo "  bash scripts/autodl_connect.sh shell         # SSH 免密登录到 AutoDL"
    echo "  bash scripts/autodl_connect.sh cmd \"<命令>\"  # 在 AutoDL 上远程执行命令"
    ;;
esac
