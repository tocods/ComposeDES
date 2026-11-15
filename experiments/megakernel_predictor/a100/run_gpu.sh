#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: run_gpu.sh calibration|heldout work-directory}"
work="$(cd "${2:?specify the A100 work directory}" && pwd)"
code="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results="$work/regions"
export MIRAGE_HOME="$work/mirage"
export PATH="$work/venv/bin:/usr/local/cuda/bin:$PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

case "$mode" in
    calibration) mpk_repetitions=24; vllm_repetitions=12 ;;
    heldout) mpk_repetitions=5; vllm_repetitions=5 ;;
    *) echo "Invalid experiment mode: $mode" >&2; exit 2 ;;
esac

mkdir -p "$results"
if [[ "$mode" == calibration && ! -f "$results/prompts.json" ]]; then
    python "$code/workloads.py" --model "$work/model" --output "$results/prompts.json"
fi
sha256sum "$results/prompts.json"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.current.sm,clocks.current.memory --format=csv
python -u "$code/benchmark_mpk.py" --model "$work/model" --build "$results/compiled" \
    --mode "$mode" --prompts "$results/prompts.json" --repetitions "$mpk_repetitions" \
    --output "$results/$mode-mpk.jsonl"
"$work/vllm-venv/bin/python" -u "$code/benchmark_vllm.py" --model "$work/model" \
    --mode "$mode" --prompts "$results/prompts.json" --repetitions "$vllm_repetitions" \
    --output "$results/$mode-vllm.jsonl"
