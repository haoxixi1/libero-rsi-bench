#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# AutoDL 远程端 (RTX 5090 / Ubuntu) 一键环境初始化脚本
# 特性:
#   1. 将数据、HuggingFace 缓存、大资产全部挂载到 /root/autodl-tmp (防止 30GB 系统盘爆满)
#   2. 自动开启 AutoDL 学术加速 + HF 国内镜像 (hf-mirror.com)
#   3. 安装 tmux, htop, nvtop, libmagickwand-dev, libegl1, libgl1 等系统依赖
#   4. 创建 libero-bench (Python 3.10) 环境并自动适配 RTX 5090 (Blackwell sm_120, PyTorch cu128)
#   5. 调用 setup_libero_plus.sh 完成 LIBERO-plus 评测环境配置，并安装 IL 训练依赖 (pyarrow, torchvision)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="/root/autodl-tmp"
ENV_NAME="${ENV_NAME:-libero-bench}"

echo "=================================================================="
echo "[1/6] 配置 AutoDL 数据盘缓存目录与国内下载加速..."
echo "=================================================================="
mkdir -p "$DATA_ROOT/hf_cache" "$DATA_ROOT/pip_cache" "$REPO_ROOT/datasets" "$REPO_ROOT/checkpoints" "$REPO_ROOT/logs"
export HF_HOME="$DATA_ROOT/hf_cache"
export PIP_CACHE_DIR="$DATA_ROOT/pip_cache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export MUJOCO_GL="egl"
export NO_PROXY="localhost,127.0.0.1,mirrors.aliyun.com,pypi.tuna.tsinghua.edu.cn,mirrors.cloud.tencent.com,hf-mirror.com"
export no_proxy="$NO_PROXY"

# 修复 AutoDL 默认 http://mirrors.aliyun.com 在代理下的兼容问题，切换为 HTTPS 清华源
mkdir -p "$HOME/.pip"
cat > "$HOME/.pip/pip.conf" <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
extra-index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com
EOF

# 启用 AutoDL 官方学术加速（若存在）
if [[ -f /etc/network_turbo ]]; then
  echo "  -> 检测到 /etc/network_turbo，正在开启 AutoDL 学术资源加速..."
  source /etc/network_turbo || true
fi

# 写入 ~/.bashrc 确保新开 tmux 窗口也生效
if ! grep -q "LIBERO_RSI_AUTODL_ENV" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<EOF

# --- LIBERO_RSI_AUTODL_ENV ---
export HF_HOME="$DATA_ROOT/hf_cache"
export PIP_CACHE_DIR="$DATA_ROOT/pip_cache"
export HF_ENDPOINT="https://hf-mirror.com"
export MUJOCO_GL="egl"
[ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1 || true
EOF
fi

echo "=================================================================="
echo "[2/6] 安装 Ubuntu 系统库与会话防断连工具 (tmux / OpenGL / EGL / ImageMagick)..."
echo "=================================================================="
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  tmux htop rsync git curl unzip \
  libmagickwand-dev libgl1 libegl1 libgles2 libglvnd0 libglx0 libosmesa6-dev \
  libglfw3 libglew-dev patchelf ffmpeg

# 配置优雅的 ~/.tmux.conf (支持鼠标滚动、大历史行数、状态栏显示时间与主机)
cat > "$HOME/.tmux.conf" <<'EOF'
set -g mouse on
set -g history-limit 100000
set -g status-bg black
set -g status-fg white
set -g status-right '#[fg=green]%Y-%m-%d %H:%M:%S#[default]'
setw -g mode-keys vi
EOF

echo "=================================================================="
echo "[3/6] 初始化 Conda 环境 ($ENV_NAME, Python 3.10)..."
echo "=================================================================="
if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/root/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "[ERROR] 未找到 Conda，请确认 AutoDL 基础镜像包含 Miniconda3。"
  exit 1
fi

if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"
PYTHON_BIN="$(which python)"
echo "  -> 当前 Python: $PYTHON_BIN"

echo "=================================================================="
echo "[4/6] 检测 GPU 架构并安装适配 RTX 5090 (Blackwell sm_120) 的 PyTorch..."
echo "=================================================================="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "Unknown")
echo "  -> 检测到 GPU: $GPU_NAME"

if "$PYTHON_BIN" -c "import torch, torchvision; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "  -> PyTorch + CUDA 已安装就绪，跳过重复下载。"
elif echo "$GPU_NAME" | grep -qiE "5090|5080|Blackwell"; then
  echo "  -> 检测到 RTX 50 系列 (Blackwell)，正在安装 PyTorch (cu128)..."
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || \
    "$PYTHON_BIN" -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
else
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install torch torchvision
fi

echo "=================================================================="
echo "[5/6] 安装 LIBERO-plus 评测环境 + IL 训练依赖..."
echo "=================================================================="
"$PYTHON_BIN" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
export LIBERO_PLUS_DIR="$DATA_ROOT/third_party/LIBERO-plus"
export PYTHON_BIN="$PYTHON_BIN"
bash "$REPO_ROOT/scripts/setup_libero_plus.sh"

# 安装 IL 数据集读取与训练所需依赖 (强制锁定 numpy<2.0 以兼容 mujoco 2.3.7)
"$PYTHON_BIN" -m pip install "numpy<2.0" pyarrow tqdm tensorboard einops

echo "=================================================================="
echo "[6/6] 运行 GPU + MuJoCo EGL 离屏渲染与全链路自检..."
echo "=================================================================="
"$PYTHON_BIN" - <<'PY'
import torch
print(f"[CHECK] PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[CHECK] Device: {torch.cuda.get_device_name(0)}, Capability: {torch.cuda.get_device_capability(0)}")
    x = torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")
    print(f"[CHECK] CUDA Tensor matmul OK: norm={x.norm().item():.2f}")
PY

"$PYTHON_BIN" "$REPO_ROOT/tests/verify_all.py"

echo "=================================================================="
echo "[ALL READY] AutoDL RTX 5090 环境初始化全部完成！"
echo "使用任务管理器启动防断连训练："
echo "  bash scripts/job_manager.sh start-all"
echo "  bash scripts/job_manager.sh status"
echo "  bash scripts/job_manager.sh logs"
echo "=================================================================="
