#!/usr/bin/env bash
# OPTIONAL second environment: ${HOME}/envs/nsbench-trtllm (override with ENV_DIR=)
#
# This is NOT run as part of the normal setup. It exists so the TensorRT-LLM backend can be
# provisioned later without disturbing the main harness environment.
#
# Why a separate venv: tensorrt_llm pins its own torch build. Installing it alongside the
# transformers stack in envs/nsbench will replace torch and can break the reference backend.
#
# Status on this machine (checked 2026-08-22):
#   * There is NO stable aarch64 tensorrt_llm release. Only 1.3.0rc* release candidates,
#     published on https://pypi.nvidia.com (plain PyPI carries an sdist stub that will not build).
#   * Wheels are cp312 / manylinux_2_39_aarch64. This host is Python 3.12 with glibc 2.39 -- a match.
#   * The NGC container route (NVIDIA's sanctioned path for DGX Spark) is unavailable here:
#     docker requires sudo and this user is not in the `docker` group.
#   * sm_121 (GB10) support in these RC wheels is unverified. Treat a successful install as the
#     start of validation, not the end of it.
#
# Caveat for profiling: TensorRT-LLM executes fused kernels behind CUDA graphs. Kernel-level ncu
# metrics still work (and nsys has --cuda-graph-trace=node), but Python-level NVTX ranges cannot be
# placed at per-layer granularity. Expect engine-level and kernel-level scoping only.

set -euo pipefail

ENV_DIR="${ENV_DIR:-${HOME}/envs/nsbench-trtllm}"
PYTHON="${PYTHON:-python3.12}"
TRTLLM_VERSION="${TRTLLM_VERSION:-1.3.0rc24}"
NVIDIA_INDEX="${NVIDIA_INDEX:-https://pypi.nvidia.com}"

cat <<EOF
This will install TensorRT-LLM ${TRTLLM_VERSION} (a RELEASE CANDIDATE) into ${ENV_DIR}.
The wheel is multiple GB and pins its own torch build.
EOF
read -r -p "Continue? [y/N] " reply
[[ "${reply}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

echo "==> Creating venv at ${ENV_DIR}"
[[ -d "${ENV_DIR}" ]] || "${PYTHON}" -m venv "${ENV_DIR}"

PIP="${ENV_DIR}/bin/pip"
PY="${ENV_DIR}/bin/python"

"${PIP}" install --upgrade pip setuptools wheel

echo "==> Installing tensorrt_llm==${TRTLLM_VERSION} from ${NVIDIA_INDEX}"
"${PIP}" install \
    --extra-index-url "${NVIDIA_INDEX}" \
    "tensorrt_llm==${TRTLLM_VERSION}"

echo
echo "==> Validating the install against this GPU"
"${PY}" - <<'PYEOF'
import sys
try:
    import tensorrt_llm
    print(f"    tensorrt_llm {tensorrt_llm.__version__}")
except Exception as exc:                                    # noqa: BLE001
    print(f"    IMPORT FAILED: {exc}")
    sys.exit(1)
try:
    import torch
    print(f"    torch        {torch.__version__} (cuda {torch.version.cuda})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"    device       {p.name} sm_{p.major}{p.minor}")
        if (p.major, p.minor) == (12, 1):
            print("    NOTE: sm_121 (GB10). Confirm this TRT-LLM build ships sm_121 kernels")
            print("          before trusting any engine build.")
    else:
        print("    WARNING: torch.cuda unavailable in this env")
except Exception as exc:                                    # noqa: BLE001
    print(f"    torch check failed: {exc}")
PYEOF

echo
echo "==> Done. Point the harness at it with:  nsbench run --backend trtllm --python ${ENV_DIR}/bin/python"
