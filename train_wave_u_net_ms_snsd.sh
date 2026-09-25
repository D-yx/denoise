#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

gpu_count="${NPROC_PER_NODE:-${GPU_COUNT:-2}}"
selection_samples="${GPU_SELECTION_SAMPLES:-5}"
selection_interval="${GPU_SELECTION_INTERVAL:-1}"

mapfile -t selected_gpus < <(
    for ((sample = 1; sample <= selection_samples; sample++)); do
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits
        if ((sample < selection_samples)); then sleep "${selection_interval}"; fi
    done |
    awk -F', *' '
        { gpu=$1; util[gpu]+=$2; memory[gpu]+=100*$3/$4; count[gpu]++ }
        END {
            for (gpu in count) {
                average_util=util[gpu]/count[gpu]; average_memory=memory[gpu]/count[gpu]
                printf "%.6f %s %.1f %.1f\n", average_util+average_memory, gpu, average_util, average_memory
            }
        }' |
    sort -n | head -n "${gpu_count}" | awk '{print $2 ":" $3 ":" $4}'
)

if [[ "${#selected_gpus[@]}" -lt "${gpu_count}" || "${gpu_count}" -lt 2 ]]; then
    echo "Need at least ${gpu_count} detected GPUs for distributed training; found ${#selected_gpus[@]}." >&2
    echo "Set GPU_COUNT to a value between 2 and the number of installed GPUs." >&2
    exit 1
fi

gpu_ids=""
for item in "${selected_gpus[@]}"; do
    IFS=: read -r gpu util memory <<< "${item}"
    [[ -n "${gpu_ids}" ]] && gpu_ids+=","
    gpu_ids+="${gpu}"
    echo "Selected GPU ${gpu}: utilization=${util}%, memory=${memory}%"
done

cache_marker=".prepared/ms-snsd-long/train/.complete"
if [[ ! -f "${cache_marker}" ]] || ! grep -qx 'recipe=in-domain-speaker-noise-disjoint-v5' "${cache_marker}"; then
    echo "Prepared in-domain MS-SNSD v5 data not found. Rerun preprocessing with --force." >&2
    exit 1
fi

exec env CUDA_VISIBLE_DEVICES="${gpu_ids}" conda run --no-capture-output -n denoise \
    torchrun --standalone --nproc-per-node="${gpu_count}" \
    wave_u_net_ms_snsd.py --stage train "$@"
