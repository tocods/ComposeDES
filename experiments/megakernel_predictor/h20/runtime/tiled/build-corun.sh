#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/tools/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/build/bin:/usr/local/cuda/lib64"
python3 - <<'PY'
from pathlib import Path
base = Path("tiled/profile.cu").read_text()
base = base.replace('#include "resident-moe.cu"', '#include "resident-moe-tiled.cu"')
index = base.index("template<int batch>\nvoid measure")
base = base[:index] + Path("tiled/corun.inc").read_text() + "\n" + base[index:]
start = base.index("    for (int size : {256, 512})")
end = base.index("} // namespace", start)
body = """    auto *other = allocate<block_q8_1>(slots * ff / warp);
    for (int producers : {26, 39, 52}) {
        for (int gate_rows : {32, 64}) {
            for (int down_rows : {64, 128}) {
                const double gate = time_component<batch, true>(s, t, producers, gate_rows, 512);
                const double down = time_component<batch, false>(s, t, nsm - producers, down_rows, 512);
                const double pair = t.measure([&] {
                    paired_tiles<batch><<<nsm, 512, 0, t.stream>>>(s, other, producers, gate_rows, down_rows);
                });
                std::printf("%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.6f\\n", frame.batch, frame.layer,
                            frame.cohort, frame.step, producers, gate_rows, down_rows, loads.c_str(),
                            gate, down, pair, plain);
            }
        }
    }
    checked(cudaFree(other));
}
"""
base = base[:start] + body + base[end:]
base = base.replace("batch,layer,cohort,step,stage,blocks,threads,rows,type,expert_bytes,loads,us,plain_us,coop_us,barrier_us,setup_us,reduce_us",
                    "batch,layer,cohort,step,producers,gate_rows,down_rows,loads,gate_us,down_us,pair_us,plain_us")
Path("tiled/corun.cu").write_text(base)
PY
python3 tiled/generate.py resident-moe.cu tiled/resident-moe-tiled.cu
nvcc -std=c++17 -O3 -arch=sm_90 -DQWEN_RESIDENT_STANDALONE \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -I llama.cpp/ggml/include -I llama.cpp/ggml/src \
    -I llama.cpp/ggml/src/ggml-cuda tiled/corun.cu -L build/bin -lggml-base -o tiled/corun
