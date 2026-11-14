#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
phase="$1"
export LD_LIBRARY_PATH="$PWD/build/bin:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
export QWEN_RESIDENT_EMBEDDING=1 QWEN_RESIDENT_FIXED_ROUTE=1
export QWEN_CTA_FIRST=512 QWEN_CTA_SECOND=512
export QWEN_GATE_ROWS=32 QWEN_DOWN_ROWS=64 QWEN_MATCH_PARTITION=0
model=Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf
dst="compose-h20/$phase"
mkdir -p "$dst"
if [[ "$phase" == train ]]; then
    prompts=train-prompts.txt
    tokens=entry-profile/train/b2.tokens
    requests=8 steps=24
    orders=("stock tiled_whole split_all ready_26 ready_39 ready_52")
elif [[ "$phase" == heldout || "$phase" == unseen ]]; then
    if [[ "$phase" == unseen ]]; then
        (cd compose-h20 && sha256sum -c b2-unseen-forecast.sha256)
        prompts=compose-h20/unseen-prompts.txt
        tokens=compose-h20/unseen-b2.tokens
    else
        (cd compose-h20 && sha256sum -c b2-forecast.sha256)
        prompts=compose-h20/heldout-prompts.txt
        tokens=compose-h20/new-b2.tokens
    fi
    requests=16 steps=24
    orders=("ready_26 stock ready_39 tiled_whole ready_52 split_all"
            "split_all ready_52 tiled_whole ready_39 stock ready_26"
            "ready_39 split_all ready_26 ready_52 stock tiled_whole")
else
    exit 2
fi
for round in "${!orders[@]}"; do
    for label in ${orders[$round]}; do
        case "$label" in
            stock) cut=0; producers=0 ;;
            tiled_whole) cut=7; producers=0 ;;
            split_all) cut=9; producers=26 ;;
            ready_*) cut=8; producers="${label#ready_}" ;;
        esac
        flags=()
        if [[ "$phase" == train || "$label" == stock ]]; then
            flags+=(QWEN_SKIP_REFERENCE=1)
        fi
        env "${flags[@]}" QWEN_RESIDENT_CUT="$cut" QWEN_PRODUCER_CTAS="$producers" \
            ./bench "$model" "$prompts" 2 "$requests" "$steps" "$tokens" \
            > "$dst/$round-$label.csv.part" 2> "$dst/$round-$label.log"
        mv "$dst/$round-$label.csv.part" "$dst/$round-$label.csv"
    done
done
