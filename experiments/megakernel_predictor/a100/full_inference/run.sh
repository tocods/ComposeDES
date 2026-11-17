#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
NVCC=${NVCC:-nvcc}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3-8B}
OUT=${OUT:-../../runs/full-qwen}
export TORCH_CUDA_ARCH_LIST=8.0 FLASHINFER_CUDA_ARCH_LIST=8.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$OUT"
nvidia-smi -q > "$OUT/gpu-health.txt"
extra=()
if [[ ${PERFORMANCE_ONLY:-0} == 1 ]]; then
  extra+=(--performance-only)
  echo 'Performance-only mode: numerical correctness and memory-health checks do not gate timing.'
else
  if grep -Eq 'Remapping Failure Occurred[[:space:]]*:[[:space:]]*Yes' "$OUT/gpu-health.txt"; then
    echo 'GPU reports memory remapping failure; request platform diagnosis before benchmarking.' >&2
    exit 1
  fi
  "$NVCC" -O3 -arch=sm_80 memory_check.cu -o "$OUT/memory_check"
  for trial in 1 2 3; do
    "$OUT/memory_check" > "$OUT/memory-check-$trial.txt" 2>&1
  done
fi
"$NVCC" -std=c++17 -O3 -arch=sm_80 -shared -Xcompiler -fPIC runtime.cu -lcuda -o libfull_graph.so
"$PYTHON" bench.py --model "$MODEL" --out "$OUT/smoke.jsonl" --smoke \
  --prefix 512 --suffix 64 --tokens 4 --repeat 2 --generate-repeat 1 "${extra[@]}"
"$PYTHON" suite.py --model "$MODEL" --out "$OUT" --phase calibrate "${extra[@]}"
"$PYTHON" suite.py --model "$MODEL" --out "$OUT" --phase validate "${extra[@]}"
