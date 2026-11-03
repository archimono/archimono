#!/bin/bash
#SBATCH --job-name=certify-n36
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=cluster_long
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/certify-n36-%x-%j.out
#SBATCH --error=logs/certify-n36-%x-%j.err

# Usage: sbatch scripts/slurm_certify_n36.sh <tiling_key>
#   e.g. sbatch scripts/slurm_certify_n36.sh rhombitrihexagonal
#        sbatch scripts/slurm_certify_n36.sh snub_square
#        sbatch scripts/slurm_certify_n36.sh snub_hexagonal

set -euo pipefail

TILING="${1:?Usage: sbatch slurm_certify_n36.sh <tiling_key>}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JSON_OUT="${REPO_ROOT}/tmp/bruteforce-certification/${TILING}-n36"
MD_OUT="${REPO_ROOT}/tmp/bruteforce-certification/${TILING}-n36.md"

mkdir -p "${JSON_OUT}"

echo "tiling=${TILING}"
echo "json_out=${JSON_OUT}"
echo "md_out=${MD_OUT}"
echo "cpus=${SLURM_CPUS_PER_TASK}"
date

CONDA_BASE=${CONDA_BASE:-}
if [[ -z "${CONDA_BASE}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base)
    elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        CONDA_BASE=$(dirname "$(dirname "${CONDA_EXE}")")
    else
        for candidate in \
            "/work/${USER}/miniconda3" \
            "/work/${USER}/anaconda3" \
            "${HOME}/miniconda3" \
            "${HOME}/anaconda3" \
            "${HOME}/miniforge3"; do
            if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
                CONDA_BASE=${candidate}
                break
            fi
        done
    fi
fi
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: could not locate conda. Set CONDA_BASE=/path/to/miniconda3." >&2
    exit 2
fi
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate archimono

export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"

archimono certify \
    --solver bruteforce \
    --tilings "${TILING}" \
    --target-sizes 36 \
    --json-out-dir "${JSON_OUT}" \
    --markdown-out "${MD_OUT}"

echo "done"
date
