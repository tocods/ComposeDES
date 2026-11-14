#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
h20="$(dirname "$here")"
predictor="$(dirname "$h20")"
dest="$(mkdir -p "$1" && cd "$1" && pwd)"
revision=51e5c9afbbccb353cefd698036ab9ed00a3612ab

for file in bench.cpp capture.cpp resident-moe.cu resident-moe.cuh stage_profile.cu model.sha256 train_prompts.py llama-experiment.patch; do
    cp "$here/$file" "$dest/"
done
mkdir -p "$dest"/{tiled,tuning,compose-h20,entry-profile/train,dev}
cp "$here"/tiled/{*.inc,*.py,*.sh} "$dest/tiled/"
cp "$here"/tuning/{*.inc,*.py} "$dest/tuning/"
cp "$here"/compose-h20/*.tokens "$dest/compose-h20/"
cp "$here"/entry-profile/train/b2.tokens "$dest/entry-profile/train/"
cp "$h20"/{build_trace.py,trace.inc,fp8_full.py,fp8_components.py,run_fp8.sh,run_b2.sh,heldout-prompts.txt,unseen-prompts.txt} "$dest/compose-h20/"
cp "$predictor"/{contention.py,export_routes.py} "$dest/compose-h20/"

if [[ ! -d "$dest/pristine/.git" ]]; then
    git init "$dest/pristine"
    git -C "$dest/pristine" remote add origin https://github.com/unixsysdev/llama-turboquant.git
    git -C "$dest/pristine" fetch --depth 1 origin "$revision"
    git -C "$dest/pristine" checkout --detach FETCH_HEAD
fi
test "$(git -C "$dest/pristine" rev-parse HEAD)" = "$revision"
if [[ ! -d "$dest/llama.cpp" ]]; then
    mkdir "$dest/llama.cpp"
    git -C "$dest/pristine" archive "$revision" | tar -xf - -C "$dest/llama.cpp"
    (cd "$dest/llama.cpp" && git apply ../llama-experiment.patch)
fi
if [[ "${2:-}" == "--prepare-only" ]]; then
    exit 0
fi

cd "$dest"
cuda="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$cuda/bin:$PATH"
flags=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90
       -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF
       -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF)
for item in experimental stock; do
    case "$item" in
        experimental) tree=llama.cpp; out=build; executable=bench;;
        stock) tree=pristine; out=pristine/build; executable=bench-stock;;
    esac
    cmake -S "$tree" -B "$out" "${flags[@]}"
    cmake --build "$out" --target llama -j "${BUILD_JOBS:-12}"
    for source in bench.cpp capture.cpp; do
        if [[ "$source" == capture.cpp ]]; then
            [[ "$item" == experimental ]] || continue
            executable=capture
        fi
        c++ -std=c++17 -O3 "$source" -I "$tree/include" -I "$tree/ggml/include" -I "$cuda/include" \
            -L "$out/bin" -Wl,-rpath,"$PWD/$out/bin" -L "$cuda/lib64" -Wl,-rpath,"$cuda/lib64" \
            -lllama -lggml -lggml-base -lcudart -ldl -o "$executable"
    done
done
python3 compose-h20/build_trace.py . compose-h20
nvcc -std=c++17 -O3 -arch=sm_90 -DQWEN_RESIDENT_STANDALONE -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -I . -I llama.cpp/ggml/include -I llama.cpp/ggml/src -I llama.cpp/ggml/src/ggml-cuda \
    compose-h20/tile_trace.cu -L build/bin -lggml-base -o compose-h20/tile_trace_v2
