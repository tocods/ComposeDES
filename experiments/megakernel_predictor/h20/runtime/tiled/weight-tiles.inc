// Gate/up tiles cover disjoint ranges of complete 32-value quantization groups.
template<bool first, int batch, int type>
__device__ __forceinline__ void tile_project(State s, const Group &g, int start,
                                             int width, float *scratch) {
    constexpr int cols = first ? hidden : ff;
    const int row_bytes = first ? s.first_bytes / ff : s.down_bytes / hidden;
    const int lane = threadIdx.x % warp;
    for (int local = threadIdx.x / warp; local < width; local += blockDim.x / warp) {
        const int row = start + local;
        const auto *a = first ? s.gate + g.expert * s.first_bytes + row * row_bytes
                              : s.down + g.expert * s.down_bytes + row * row_bytes;
        const auto *b = s.up + g.expert * s.first_bytes + row * row_bytes;
        float sa[batch] = {}, sb[batch] = {};
        for (int col = lane; col < cols / warp; col += warp) {
#pragma unroll
            for (int j = 0; j < batch; ++j) {
                if (j < g.count) {
                    const int id = g.routes[j];
                    const auto *x = first ? s.xq + id / top * (hidden / warp)
                                          : s.hq + id * (ff / warp);
                    sa[j] += dot<type>(a, x, col);
                    if constexpr (first) sb[j] += dot<type>(b, x, col);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < batch; ++j) {
            if (j < g.count) {
                const float ga = sum_warp(sa[j]);
                float up = 0;
                if constexpr (first) up = sum_warp(sb[j]);
                if (lane == 0) {
                    if constexpr (first) scratch[j * width + local] = up * (ga / (1.0f + expf(-ga)));
                    else s.down_output[g.routes[j] * hidden + row] = ga;
                }
            }
        }
    }
}

template<bool first, int batch>
__device__ __forceinline__ void tile_project(State s, const Group &group, int start,
                                             int width, float *scratch) {
    switch (first ? s.first_type : s.down_type) {
        case iq2: tile_project<first, batch, iq2>(s, group, start, width, scratch); break;
        case iq3: tile_project<first, batch, iq3>(s, group, start, width, scratch); break;
        case iq4: tile_project<first, batch, iq4>(s, group, start, width, scratch); break;
    }
}

using TileReady = cuda::atomic_ref<unsigned int, cuda::thread_scope_device>;

template<int batch, bool publish>
__device__ __forceinline__ void gate_tiles(State s, int worker, int workers, int width) {
    __shared__ float scratch[batch * 128];
    const int count = ff / width;
    const int lane = threadIdx.x % warp;
    for (int task = worker; task < s.routes->used * count; task += workers) {
        const int index = task / count, offset = task % count * width;
        const auto &group = s.routes->groups[index];
        tile_project<true, batch>(s, group, offset, width, scratch);
        __syncthreads();
        const int groups = width / warp;
        for (int i = threadIdx.x / warp; i < group.count * groups; i += blockDim.x / warp) {
            const float x = scratch[i * warp + lane];
            const float max = warp_reduce_max<warp>(fabsf(x));
            const float sum = warp_reduce_sum<warp>(x);
            const float scale = max / 127.f;
            auto &out = s.hq[group.routes[i / groups] * (ff / warp) + offset / warp + i % groups];
            out.qs[lane] = max == 0 ? 0 : roundf(x / scale);
            if (!lane) out.ds = make_half2(scale, sum);
        }
        // Ensure the complete tile is visible before publishing or reusing shared memory.
        __syncthreads();
        if constexpr (publish) {
            if (threadIdx.x == 0) TileReady(s.routes->ready[index]).fetch_add(1, cuda::memory_order_acq_rel);
        }
    }
}

template<int batch, bool wait_ready>
__device__ __forceinline__ void down_tiles(State s, int worker, int workers, int width, int gate_width) {
    const int count = hidden / width;
    for (int task = worker; task < s.routes->used * count; task += workers) {
        const int index = task / count;
        if constexpr (wait_ready) {
            if (threadIdx.x == 0) {
                while (TileReady(s.routes->ready[index]).load(cuda::memory_order_acquire) != ff / gate_width)
                    __nanosleep(64);
            }
            __syncthreads();
        }
        tile_project<false, batch>(s, s.routes->groups[index], task % count * width, width, nullptr);
    }
}

template<int batch, bool ready>
__global__ void tiled_fusion(State s, int producers, int gate_width, int down_width) {
    const auto grid = cg::this_grid();
    route(s);
    quantize<hidden>(s.activation, s.xq, s.batch);
    grid.sync();
    if (producers == 0 || blockIdx.x < producers)
        gate_tiles<batch, ready>(s, blockIdx.x, producers == 0 ? gridDim.x : producers, gate_width);
    if constexpr (!ready) grid.sync();
    if (blockIdx.x >= producers)
        down_tiles<batch, ready>(s, blockIdx.x - producers, gridDim.x - producers, down_width, gate_width);
    grid.sync();
    reduce(s);
}

template<int batch>
__global__ void tiled_gate(State s, int gate_width) {
    const auto grid = cg::this_grid();
    route(s);
    quantize<hidden>(s.activation, s.xq, s.batch);
    grid.sync();
    gate_tiles<batch, false>(s, blockIdx.x, gridDim.x, gate_width);
}

template<int batch>
__global__ void tiled_down(State s, int down_width) {
    down_tiles<batch, false>(s, blockIdx.x, gridDim.x, down_width, 0);
    cg::this_grid().sync();
    reduce(s);
}

template<int batch>
void launch_tiles(cudaStream_t stream, int blocks, State &s, int mode, int first, int second) {
    static const auto option = [](const char *key, int fallback) {
        const char *value = std::getenv(key);
        return value ? std::atoi(value) : fallback;
    };
    static int gate_width = option("QWEN_GATE_ROWS", 64);
    static int down_width = option("QWEN_DOWN_ROWS", 128);
    static int partition = option("QWEN_PRODUCER_CTAS", 26);
    GGML_ASSERT(gate_width == 32 || gate_width == 64 || gate_width == 128);
    GGML_ASSERT(down_width == 64 || down_width == 128 || down_width == 256 || down_width == 512);
    GGML_ASSERT((partition > 0 || (partition == 0 && mode == 7)) && partition < blocks);
    if (mode == 9) {
        // The split path can use all SMs, or use the same partition as fusion.
        static const bool matched = option("QWEN_MATCH_PARTITION", 0) != 0;
        void *gate[] = {&s, &gate_width};
        void *down[] = {&s, &down_width};
        CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(tiled_gate<batch>),
                   matched ? partition : blocks, first, gate, 0, stream));
        CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(tiled_down<batch>),
                   matched ? blocks - partition : blocks, second, down, 0, stream));
        return;
    }
    void *args[] = {&s, &partition, &gate_width, &down_width};
    const void *kernel = mode == 8
        ? reinterpret_cast<const void *>(tiled_fusion<batch, true>)
        : reinterpret_cast<const void *>(tiled_fusion<batch, false>);
    CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, blocks, first, args, 0, stream));
}
