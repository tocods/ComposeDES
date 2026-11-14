#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PWD/tools/bin:$PATH"
export PYTHONPATH="$PWD/compose-h20/fp8-python:$PWD/specialized/python:$PWD/specialized/python/nvidia_cutlass_dsl/dsl_packages:$PWD/tools:${PYTHONPATH:-}"
export FLASHINFER_CUDA_ARCH_LIST=9.0a
phase="$1"
backend="${2:-eager}"
flags=()
if [[ "$backend" == graph ]]; then
    flags+=(--cuda-graph)
elif [[ "$backend" != eager ]]; then
    exit 2
fi
dir="compose-h20/fp8-$phase-$backend"
mkdir -p "$dir"
case "$phase" in
    train)
        head -n 8 train-prompts.txt > "$dir/prompts.txt"
        tokens=entry-profile/train/b2.tokens
        rounds=("flags barrier split_cached")
        ;;
    heldout)
        (cd compose-h20 && sha256sum -c "fp8-forecast-$backend.sha256")
        cp compose-h20/unseen-prompts.txt "$dir/prompts.txt"
        tokens=compose-h20/unseen-b2.tokens
        python compose-h20/fp8_full.py compose-h20/model "$dir/prompts.txt" \
            --tiled tiled --teacher-gguf "$tokens" --campaign-dir "$dir" "${flags[@]}"
        exit 0
        ;;
    *) exit 2 ;;
esac
for round in "${!rounds[@]}"; do
    for variant in ${rounds[$round]}; do
        stem="$dir/$round-$variant"
        QWEN_MONO_BOUNDARY="$variant" python compose-h20/fp8_full.py compose-h20/model "$dir/prompts.txt" \
            --tiled tiled --teacher-gguf "$tokens" --outputs "$stem.out.json" "${flags[@]}" \
            > "$stem.csv.part" 2> "$stem.log"
        mv "$stem.csv.part" "$stem.csv"
    done
done
