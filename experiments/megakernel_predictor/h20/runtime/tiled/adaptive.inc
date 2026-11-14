// A finished producer can begin down work without reassigning any resident CTA
// that still owns gate tiles. Consumers only claim work whose data is ready.
template<int batch>
__device__ __forceinline__ void claim_down(State s, int width, int gate_width) {
    const int tasks = hidden / width;
    const int used = s.routes->used;
    int cursor = blockIdx.x % used;
    __shared__ int selected_expert, selected_row;
    for (;;) {
        if (threadIdx.x == 0) {
            selected_expert = -1;
            while (TileReady(s.routes->unclaimed).load(cuda::memory_order_relaxed) != 0) {
                const int e = cursor;
                cursor = cursor + 1 == used ? 0 : cursor + 1;
                TileReady next(s.routes->next_down[e]);
                unsigned int i = next.load(cuda::memory_order_relaxed);
                if (i >= tasks ||
                    TileReady(s.routes->ready[e]).load(cuda::memory_order_acquire) != ff / gate_width) {
                    __nanosleep(16);
                    continue;
                }
                constexpr unsigned int claim_size = 2;
                if (next.compare_exchange_strong(i, i + claim_size, cuda::memory_order_relaxed)) {
                    selected_expert = e;
                    selected_row = i * width;
                    TileReady(s.routes->unclaimed).fetch_sub(claim_size, cuda::memory_order_relaxed);
                    break;
                }
            }
        }
        __syncthreads();
        const int e = selected_expert, row = selected_row;
        if (e < 0) return;
        tile_project<false, batch>(s, s.routes->groups[e], row, width * 2, nullptr);
        __syncthreads();
    }
}

template<int batch, bool transfer>
__global__ void tiled_queued(State s, int producers, int gate_width, int down_width) {
    const auto grid = cg::this_grid();
    route(s);
    if (blockIdx.x == 0 && threadIdx.x == 0)
        s.routes->unclaimed = s.routes->used * (hidden / down_width);
    quantize<hidden>(s.activation, s.xq, s.batch);
    grid.sync();
    if (blockIdx.x < producers) gate_tiles<batch, true>(s, blockIdx.x, producers, gate_width);
    if (transfer || blockIdx.x >= producers) claim_down<batch>(s, down_width, gate_width);
    grid.sync();
    reduce(s);
}

template<int batch>
void launch_queued(cudaStream_t stream, int blocks, State &s, int mode, int first) {
    static int gate_rows = std::getenv("QWEN_GATE_ROWS") ? std::atoi(std::getenv("QWEN_GATE_ROWS")) : 32;
    static int down_rows = std::getenv("QWEN_DOWN_ROWS") ? std::atoi(std::getenv("QWEN_DOWN_ROWS")) : 64;
    static int producers = std::getenv("QWEN_PRODUCER_CTAS") ? std::atoi(std::getenv("QWEN_PRODUCER_CTAS")) : 26;
    GGML_ASSERT(gate_rows == 32 || gate_rows == 64 || gate_rows == 128);
    GGML_ASSERT(down_rows == 64 || down_rows == 128 || down_rows == 256 || down_rows == 512);
    GGML_ASSERT(producers > 0 && producers < blocks);
    void *args[] = {&s, &producers, &gate_rows, &down_rows};
    const void *kernel = mode == 10
        ? reinterpret_cast<const void *>(tiled_queued<batch, true>)
        : reinterpret_cast<const void *>(tiled_queued<batch, false>);
    CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, blocks, first, args, 0, stream));
}
