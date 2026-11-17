#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
NVCC=${NVCC:-nvcc}
OUT=${OUT:-../../runs/graph-edges}
export OMP_NUM_THREADS=1
mkdir -p "$OUT"
"$NVCC" -O3 -std=c++17 -arch=sm_80 -shared -Xcompiler -fPIC graph_runtime.cu -o libgraph_runtime.so
"$PYTHON" verify.py
"$PYTHON" experiment.py --out "$OUT/cal-small-A.jsonl" --n 1024 --rounds 32 --candidates 48 --repeat 16
"$PYTHON" experiment.py --out "$OUT/cal-medium-A.jsonl" --seed 12 --n 4096 --rounds 256 --candidates 48 --repeat 16
"$PYTHON" experiment.py --out "$OUT/cal-heavy-A.jsonl" --seed 13 --n 4096 --rounds 1024 --candidates 48 --repeat 16
"$PYTHON" experiment.py --out "$OUT/cal-wide-A.jsonl" --seed 14 --shape fork --n 16384 --rounds 128 --candidates 48 --repeat 16
"$PYTHON" learn.py --out "$OUT/gcn-dqn.pt" --epochs 100
"$PYTHON" predict.py freeze --model "$OUT/frozen-des.json" \
  "$OUT/cal-small-A.jsonl" "$OUT/cal-medium-A.jsonl" "$OUT/cal-heavy-A.jsonl" "$OUT/cal-wide-A.jsonl"
"$PYTHON" validate.py prepare "$OUT"
"$PYTHON" validate.py run "$OUT"
"$PYTHON" analyze.py "$OUT"
