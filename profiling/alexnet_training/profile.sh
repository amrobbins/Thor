#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${THOR_PYTHON:-${REPO_ROOT}/.venv/bin/python3}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="${THOR_PYTHON:-$(command -v python3 || true)}"
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "Could not find Python. Set THOR_PYTHON to the Thor venv's python3." >&2
    exit 2
fi

# Profiling from a source checkout must exercise the just-built Thor, not an
# installed thor-cuda/kernel-wheel pair that may describe a different source
# revision.  This is the same source-tree import mode used by check-python.
THOR_PYTHON_BUILD_DIR="${THOR_PYTHON_BUILD_DIR:-${REPO_ROOT}/build/bindings/python}"
if ! compgen -G "${THOR_PYTHON_BUILD_DIR}/thor/_thor*.so" >/dev/null; then
    cat >&2 <<EOF_BUILD
Could not find the built Thor Python extension under:
  ${THOR_PYTHON_BUILD_DIR}/thor/

Build the current checkout before profiling, for example:
  cmake --build "${REPO_ROOT}/build" --target _thor -j

Set THOR_PYTHON_BUILD_DIR only if this checkout uses a different CMake build directory.
EOF_BUILD
    exit 2
fi
if [[ ! -f "${THOR_PYTHON_BUILD_DIR}/thor/__init__.py" ]]; then
    cat >&2 <<EOF_BUILD
The Thor build-tree Python package is incomplete under:
  ${THOR_PYTHON_BUILD_DIR}/thor/

Rebuild the current checkout before profiling, for example:
  cmake --build "${REPO_ROOT}/build" --target _thor -j
EOF_BUILD
    exit 2
fi

export THOR_CUDA_BOOTSTRAP_SOURCE_TREE=1
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${THOR_PYTHON_BUILD_DIR}:${PYTHONPATH}"
else
    export PYTHONPATH="${THOR_PYTHON_BUILD_DIR}"
fi

# Fail before launching Nsight if the current build tree cannot be imported.
# In particular, this catches stale _thor/libThor combinations as an ordinary
# build problem instead of producing a confusing `No reports were generated`
# message after Nsight starts.
if ! THOR_IMPORTED_FROM="$("${PYTHON}" -c 'from pathlib import Path; import thor; print(Path(thor.__file__).resolve())')"; then
    cat >&2 <<EOF_IMPORT
Failed to import the just-built Thor Python package from:
  ${THOR_PYTHON_BUILD_DIR}

Rebuild the current checkout and retry:
  cmake --build "${REPO_ROOT}/build" --target _thor -j
EOF_IMPORT
    exit 2
fi
case "${THOR_IMPORTED_FROM}" in
    "${THOR_PYTHON_BUILD_DIR}"/thor/*) ;;
    *)
        cat >&2 <<EOF_IMPORT
Thor resolved outside the requested build tree:
  ${THOR_IMPORTED_FROM}
Expected it under:
  ${THOR_PYTHON_BUILD_DIR}/thor/

Check THOR_PYTHON, THOR_PYTHON_BUILD_DIR, and PYTHONPATH before profiling.
EOF_IMPORT
        exit 2
        ;;
esac

echo "Thor profiling: using build-tree package ${THOR_IMPORTED_FROM}" >&2

# Prefer the launcher implementation from this source checkout.  That keeps
# profiler lifecycle/finalization behavior in lockstep with the runtime being
# profiled even if the venv still contains an older thor-nsys-profile install.
# THOR_NSYS_PROFILE remains an explicit override for one-off experiments.
NSYS_LAUNCHER=()
if [[ -n "${THOR_NSYS_PROFILE:-}" ]]; then
    NSYS_LAUNCHER=("${THOR_NSYS_PROFILE}")
elif [[ -f "${REPO_ROOT}/bindings/python/src/thor_nsys_profile/__init__.py" ]]; then
    NSYS_LAUNCHER=(
        "${PYTHON}"
        -c
        'import sys; sys.path.insert(0, sys.argv[1]); from thor_nsys_profile import main; raise SystemExit(main(sys.argv[2:]))'
        "${REPO_ROOT}/bindings/python/src"
    )
elif [[ -x "${REPO_ROOT}/.venv/bin/thor-nsys-profile" ]]; then
    NSYS_LAUNCHER=("${REPO_ROOT}/.venv/bin/thor-nsys-profile")
elif command -v thor-nsys-profile >/dev/null 2>&1; then
    NSYS_LAUNCHER=("$(command -v thor-nsys-profile)")
elif "${PYTHON}" -c 'import thor_nsys_profile' >/dev/null 2>&1; then
    NSYS_LAUNCHER=("${PYTHON}" -m thor_nsys_profile)
else
    echo "Could not find Thor's Nsight launcher. Build/install the Thor Python package or set THOR_NSYS_PROFILE." >&2
    exit 2
fi

DEFAULT_OUTPUT="${REPO_ROOT}/build/profiling/alexnet_training/alexnet_trainer.nsys-rep"
OUTPUT="${THOR_ALEXNET_PROFILE_OUTPUT:-${DEFAULT_OUTPUT}}"
mkdir -p "$(dirname -- "${OUTPUT}")"

exec "${NSYS_LAUNCHER[@]}" -- \
    "${PYTHON}" "${SCRIPT_DIR}/alexnet_training_profile.py" \
    --output "${OUTPUT}" \
    "$@"
