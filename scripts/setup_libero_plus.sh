#!/usr/bin/env bash
set -euo pipefail

# One-click installer for the sylvestf/LIBERO-plus evaluation backend.
# Intended to run inside a Python 3.10 evaluation environment (Linux GPU / Ubuntu).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBERO_PLUS_DIR="${LIBERO_PLUS_DIR:-$REPO_ROOT/third_party/LIBERO-plus}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_DIR="${LIBERO_CONFIG_PATH:-$HOME/.libero}"

echo "[1/5] Checking environment (REPO_ROOT=$REPO_ROOT, PYTHON_BIN=$PYTHON_BIN)..."
if [[ "$(uname -s)" == "Linux" ]]; then
  if command -v dpkg >/dev/null 2>&1 && ! dpkg -s libmagickwand-dev >/dev/null 2>&1; then
    echo "[WARN] Ubuntu/Debian package 'libmagickwand-dev' was not detected (needed by wand)."
    echo "       If pip fails building wand, run: sudo apt-get update && sudo apt-get install -y libmagickwand-dev libgl1 libegl1"
  fi
fi

# Remove original LIBERO if present so `import libero` always resolves to LIBERO-plus.
"$PYTHON_BIN" -m pip uninstall -y libero >/dev/null 2>&1 || true

echo "[2/5] Cloning sylvestf/LIBERO-plus into $LIBERO_PLUS_DIR..."
if [[ ! -d "$LIBERO_PLUS_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LIBERO_PLUS_DIR")"
  git clone --depth 1 https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_DIR"
else
  echo "  -> Existing git repo found at $LIBERO_PLUS_DIR, skipping clone."
fi

echo "[3/5] Installing Python dependencies and LIBERO-plus (editable)..."
"$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements-harness.txt" -r "$LIBERO_PLUS_DIR/extra_requirements.txt"
"$PYTHON_BIN" -m pip install --no-deps -e "$LIBERO_PLUS_DIR"

ASSETS_PARENT="$LIBERO_PLUS_DIR/libero/libero"
echo "[4/5] Ensuring LIBERO-plus simulator assets under $ASSETS_PARENT/assets..."
if [[ ! -d "$ASSETS_PARENT/assets/scenes" ]]; then
  "$PYTHON_BIN" - "$ASSETS_PARENT" <<'PY'
import sys
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

target_parent = Path(sys.argv[1])
target_parent.mkdir(parents=True, exist_ok=True)
zip_path = hf_hub_download(
    repo_id="Sylvest/LIBERO-plus",
    filename="assets.zip",
    repo_type="dataset",
)
print(f"  -> Extracting {zip_path} to {target_parent}...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(target_parent)
assert (target_parent / "assets" / "scenes").is_dir(), f"Missing assets/scenes under {target_parent}"
PY
else
  echo "  -> assets/scenes already exists, skipping download."
fi

echo "[5/5] Writing non-interactive LIBERO config at $CONFIG_DIR/config.yaml & verifying..."
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.yaml" <<EOF
assets: $ASSETS_PARENT/assets
bddl_files: $ASSETS_PARENT/bddl_files
benchmark_root: $ASSETS_PARENT
datasets: $ASSETS_PARENT/../datasets
init_states: $ASSETS_PARENT/init_files
EOF

LIBERO_CONFIG_PATH="$CONFIG_DIR" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

assert OffScreenRenderEnv is not None
assert "libero_spatial" in benchmark.get_benchmark_dict()
assets = Path(get_libero_path("assets"))
bddl = Path(get_libero_path("bddl_files"))
inits = Path(get_libero_path("init_states"))
tc = Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
assert (assets / "scenes").is_dir(), f"Missing scenes dir: {assets}"
assert bddl.is_dir(), f"Missing bddl dir: {bddl}"
assert inits.is_dir(), f"Missing init_states dir: {inits}"
assert tc.is_file(), f"Missing task_classification.json: {tc}"
print("[OK] LIBERO-plus environment is ready!")
print(f"  assets:         {assets}")
print(f"  bddl_files:     {bddl}")
print(f"  init_states:    {inits}")
print(f"  classification: {tc}")
PY
