#!/usr/bin/env bash
# Provision the main harness environment: /home/sarcs/envs/samarthamp
#
# Mirrors the known-good stack on this box (torch 2.13.0+cu130 aarch64 + transformers 5.x),
# which was verified to run bf16 matmuls on the GB10.
#
# The cu130 index is mandatory: PyPI's aarch64 torch wheel is CPU-only and will silently
# give you a torch.cuda.is_available() == False environment.

set -euo pipefail

ENV_DIR="${ENV_DIR:-/home/sarcs/envs/samarthamp}"
PYTHON="${PYTHON:-python3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating venv at ${ENV_DIR} using ${PYTHON}"
if [[ -d "${ENV_DIR}" ]]; then
    echo "    ${ENV_DIR} already exists; reusing it (pass ENV_DIR= to change)."
else
    "${PYTHON}" -m venv "${ENV_DIR}"
fi

PIP="${ENV_DIR}/bin/pip"
PY="${ENV_DIR}/bin/python"

echo "==> Upgrading pip toolchain"
"${PIP}" install --upgrade pip setuptools wheel

echo "==> Installing torch ${TORCH_VERSION} (+cu130, aarch64) from ${TORCH_INDEX}"
"${PIP}" install --index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}"

echo "==> Installing harness requirements"
"${PIP}" install -r "${HERE}/requirements.txt"

echo "==> Installing nsight_bench in editable mode"
"${PIP}" install -e "${HERE}/.."

echo
echo "==> Verifying CUDA is live"
"${PY}" - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "torch built without CUDA -- wrong index URL?"
p = torch.cuda.get_device_properties(0)
print(f"    torch        {torch.__version__} (cuda {torch.version.cuda})")
print(f"    device       {p.name}  sm_{p.major}{p.minor}  {p.multi_processor_count} SMs")
print(f"    unified mem  {p.total_memory / 1e9:.1f} GB   L2 {p.L2_cache_size / 1e6:.2f} MB")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
_ = x @ x
torch.cuda.synchronize()
print("    bf16 matmul  OK")
PYEOF

echo
echo "==> Done. Activate with:  source ${ENV_DIR}/bin/activate"
echo "    Then run:            nsbench preflight"
