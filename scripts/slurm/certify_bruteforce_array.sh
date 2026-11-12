#!/bin/bash
# Complete sliced brute-force certification for one target n.
#
# Submit with sbatch (TARGET_N is the only required variable):
#
#   sbatch --export=ALL,TARGET_N=36 scripts/slurm/certify_bruteforce_array.sh
#
# CONDA_ENV defaults to "materials-informatics". CONDA_BASE is auto-detected
# from conda on PATH, $CONDA_EXE, or common install dirs; set it explicitly only
# if detection fails, e.g. CONDA_BASE=/path/to/miniconda3.
#
# This single SLURM array job:
#   1. task 0 plans all selected tilings and all valid diagonal/sheared supercells;
#   2. worker tasks scan assigned rank slices;
#   3. the last worker to see all shards present runs the reducer;
#   4. final output is certification JSON + *-labels.npz sidecars under CERT_JSON_DIR.
#
#SBATCH --job-name=bf-cert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --array=0-499
#SBATCH --output=bf-cert-%A_%a.out

set -euo pipefail

if [[ -z "${TARGET_N:-}" ]]; then
    echo "ERROR: set TARGET_N, e.g. sbatch --export=ALL,TARGET_N=36 scripts/slurm/certify_bruteforce_array.sh" >&2
    exit 2
fi

TILINGS=${TILINGS:-all}
MAX_STATES_PER_SLICE=${MAX_STATES_PER_SLICE:-1000000000}
MAX_SLICES=${MAX_SLICES:-64}
CONDA_ENV=${CONDA_ENV:-materials-informatics}
CONDA_BASE=${CONDA_BASE:-}
WORKER_CPUS=${SLURM_CPUS_PER_TASK:-48}
MAX_ARRAY_TASKS=500
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}"

JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
RUN_DIR=${RUN_DIR:-bruteforce-certification/n${TARGET_N}/${JOB_ID}}
MANIFEST=${MANIFEST:-${RUN_DIR}/manifest.json}
SHARD_DIR=${SHARD_DIR:-${RUN_DIR}/shards}
READY_FILE=${RUN_DIR}/manifest.ready
ERROR_FILE=${RUN_DIR}/manifest.error
REDUCE_LOCK=${RUN_DIR}/reduce.lock
REDUCE_DONE=${RUN_DIR}/reduce.done
CERT_JSON_DIR=${CERT_JSON_DIR:-tmp/bruteforce-certification-hpc/n${TARGET_N}}

cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}" "${SHARD_DIR}" "$(dirname "${CERT_JSON_DIR}")"

# Resolve a conda base directory containing etc/profile.d/conda.sh.
# Priority: explicit CONDA_BASE, conda on PATH, $CONDA_EXE (exported by conda
# init and carried into the job via --export=ALL), then common install dirs.
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

if [[ -z "${CONDA_BASE}" ]]; then
    echo "ERROR: could not locate a conda installation. Set CONDA_BASE=/path/to/miniconda3." >&2
    exit 2
fi
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: CONDA_BASE=${CONDA_BASE} does not contain etc/profile.d/conda.sh" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "${CONDA_ENV}"

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python is not available after activating CONDA_ENV=${CONDA_ENV}" >&2
    exit 2
fi

export NUMBA_NUM_THREADS="${WORKER_CPUS}"

if [[ "${TASK_ID}" -eq 0 ]]; then
    echo "Planning sliced brute-force certification"
    echo "  TARGET_N=${TARGET_N}"
    echo "  TILINGS=${TILINGS}"
    echo "  PROJECT_DIR=${PROJECT_DIR}"
    echo "  CONDA_BASE=${CONDA_BASE}"
    echo "  CONDA_ENV=${CONDA_ENV}"
    echo "  MANIFEST=${MANIFEST}"
    echo "  SHARD_DIR=${SHARD_DIR}"
    echo "  CERT_JSON_DIR=${CERT_JSON_DIR}"
    echo "  MAX_ARRAY_TASKS=${MAX_ARRAY_TASKS} (fixed by #SBATCH --array=0-499)"

    rm -f "${READY_FILE}" "${ERROR_FILE}" "${REDUCE_DONE}"
    rm -rf "${REDUCE_LOCK}"

    TMP_MANIFEST=${MANIFEST}.tmp
    # shellcheck disable=SC2086
    python scripts/sliced_certify.py plan \
        --tilings ${TILINGS} \
        --target-n "${TARGET_N}" \
        --max-states-per-slice "${MAX_STATES_PER_SLICE}" \
        --max-slices "${MAX_SLICES}" \
        --out "${TMP_MANIFEST}"

    read -r TOTAL_TASKS CASE_COUNT < <(python -c "import json; m=json.load(open('${TMP_MANIFEST}')); print(m['total_tasks'], len(m['cases']))")
    if [[ "${CASE_COUNT}" -gt "${MAX_ARRAY_TASKS}" ]]; then
        echo "ERROR: ${CASE_COUNT} cases exceed MAX_ARRAY_TASKS=${MAX_ARRAY_TASKS}; cannot submit at least one task per case." | tee "${ERROR_FILE}" >&2
        exit 1
    fi
    if [[ "${TOTAL_TASKS}" -gt "${MAX_ARRAY_TASKS}" ]]; then
        ADJUSTED_MAX_SLICES=$((MAX_ARRAY_TASKS / CASE_COUNT))
        if [[ "${ADJUSTED_MAX_SLICES}" -lt 1 ]]; then
            ADJUSTED_MAX_SLICES=1
        fi
        echo "Initial plan requested ${TOTAL_TASKS} worker tasks, above MAX_ARRAY_TASKS=${MAX_ARRAY_TASKS}."
        echo "Replanning with MAX_SLICES=${ADJUSTED_MAX_SLICES}."
        # shellcheck disable=SC2086
        python scripts/sliced_certify.py plan \
            --tilings ${TILINGS} \
            --target-n "${TARGET_N}" \
            --max-states-per-slice "${MAX_STATES_PER_SLICE}" \
            --max-slices "${ADJUSTED_MAX_SLICES}" \
            --out "${TMP_MANIFEST}"
        read -r TOTAL_TASKS CASE_COUNT < <(python -c "import json; m=json.load(open('${TMP_MANIFEST}')); print(m['total_tasks'], len(m['cases']))")
    fi
    if [[ "${TOTAL_TASKS}" -lt 1 ]]; then
        echo "ERROR: manifest has total_tasks=${TOTAL_TASKS}" | tee "${ERROR_FILE}" >&2
        exit 1
    fi

    mv "${TMP_MANIFEST}" "${MANIFEST}"
    echo "${TOTAL_TASKS}" > "${READY_FILE}"
    echo "Planned ${CASE_COUNT} cases -> ${TOTAL_TASKS} worker tasks."
fi

for _ in $(seq 1 600); do
    if [[ -f "${ERROR_FILE}" ]]; then
        cat "${ERROR_FILE}" >&2
        exit 1
    fi
    if [[ -f "${READY_FILE}" ]]; then
        break
    fi
    sleep 1
done

if [[ ! -f "${READY_FILE}" ]]; then
    echo "ERROR: timed out waiting for manifest from task 0." >&2
    exit 1
fi

TOTAL_TASKS=$(cat "${READY_FILE}")
if [[ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]]; then
    echo "Task ${TASK_ID} is above planned task count ${TOTAL_TASKS}; exiting."
    exit 0
fi

python scripts/sliced_certify.py run-task \
    --manifest "${MANIFEST}" \
    --task-id "${TASK_ID}" \
    --out-dir "${SHARD_DIR}"

SHARD_COUNT=$(find "${SHARD_DIR}" -name 'shard_*.json' | wc -l)
if [[ "${SHARD_COUNT}" -eq "${TOTAL_TASKS}" ]]; then
    if mkdir "${REDUCE_LOCK}" 2>/dev/null; then
        echo "All ${TOTAL_TASKS} shards are present; reducing."
        rm -rf "${CERT_JSON_DIR}"
        mkdir -p "${CERT_JSON_DIR}"
        python scripts/sliced_certify.py reduce \
            --manifest "${MANIFEST}" \
            --shard-dir "${SHARD_DIR}" \
            --certification-json-dir "${CERT_JSON_DIR}"
        date > "${REDUCE_DONE}"
        echo "Certification artifacts written to ${CERT_JSON_DIR}"
    else
        echo "All shards present; another task is reducing."
    fi
else
    echo "Task ${TASK_ID} finished. Shards present: ${SHARD_COUNT}/${TOTAL_TASKS}."
fi
