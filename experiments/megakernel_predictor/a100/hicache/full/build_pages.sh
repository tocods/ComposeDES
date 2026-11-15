#!/usr/bin/env bash
set -euo pipefail
mirage="$(realpath "${1:?path to patched Mirage}")"
flash="$(realpath "${2:?path to the FlashInfer include directory}")"
cd "$(dirname "$0")"
nvcc -arch=sm_80 -lineinfo -O3 -std=c++20 --expt-relaxed-constexpr \
  -use_fast_math -shared -Xcompiler=-fPIC \
  -I "$flash/../cccl/libcudacxx/include" -I "$flash/../cccl/cub" -I "$flash/../cccl/thrust" \
  -I "$mirage/include" -I "$mirage/include/mirage/persistent_kernel" \
  -I "$mirage/deps/cutlass/include" -I "$mirage/deps/cutlass/tools/util/include" -I "$flash" \
  -DMPK_HICACHE -DMAX_WORKER_PER_SCHEDULER=3 -DMPK_TARGET_CC=80 \
  -DMODE_ONLINE_NOTOKEN -DMIRAGE_BACKEND_USE_CUDA -DMPK_NUM_KV_GROUPS=1 \
  -DMPK_MAX_NUM_BATCHED_REQUESTS=1 -DMPK_MAX_NUM_BATCHED_TOKENS=8 \
  -DMPK_MAX_NUM_PAGES=64 -DMPK_MAX_SEQ_LENGTH=16384 \
  mpk_pages.cu -o libmpk_pages.so
