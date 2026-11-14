template<int batch>
__global__ void traced(State s, int producers, int gate_width, int down_width, int overlapped) {
    const auto grid = cg::this_grid();
    route(s);
    quantize<hidden>(s.activation, s.xq, s.batch);
    grid.sync();
    if (blockIdx.x < producers)
        gate_tiles<batch, true>(s, blockIdx.x, producers, gate_width);
    if (!overlapped) grid.sync();
    if (blockIdx.x >= producers)
        down_tiles<batch, true>(s, blockIdx.x - producers, gridDim.x - producers, down_width, gate_width);
    grid.sync();
    reduce(s);
}

__global__ void evict_l2(int *scratch, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        scratch[i] = i;
}

template<int batch>
void measure(State &s, Timings &t, int nsm, const Data &frame) {
    constexpr int gate_width = 32, down_width = 64, nthreads = 512;
    constexpr int max_tasks = slots * (ff / gate_width + hidden / down_width);
    auto *stamps = allocate<Stamp>(max_tasks);
    constexpr int cache_words = 96 * 1024 * 1024 / sizeof(int);
    static auto *cache = allocate<int>(cache_words);
    std::vector<Stamp> records(max_tasks);
    std::vector<float> expected(batch * hidden), actual(batch * hidden);
    for (int p : {26, 39, 52}) {
        for (int trial = 0; trial < 4; ++trial) {
            for (int pass = 0; pass < 2; ++pass) {
                int mode = (trial + pass) % 2;
                int gr = gate_width, dr = down_width;
                s.stamps = stamps;
                checked(cudaMemsetAsync(stamps, 0, max_tasks * sizeof(Stamp), t.stream));
                evict_l2<<<nsm * 4, 256, 0, t.stream>>>(cache, cache_words);
                void *args[] = {&s, &p, &gr, &dr, &mode};
                auto *kernel = reinterpret_cast<const void *>(traced<batch>);
                checked(cudaLaunchCooperativeKernel(kernel, nsm, nthreads, args, 0, t.stream));
                checked(cudaMemcpyAsync(records.data(), stamps, records.size() * sizeof(Stamp),
                                        cudaMemcpyDeviceToHost, t.stream));
                checked(cudaMemcpyAsync(actual.data(), s.out, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, t.stream));
                checked(cudaStreamSynchronize(t.stream));
                if (pass == 0) expected = actual;
                else if (expected != actual) throw std::runtime_error("overlap changed output");
                for (int i = 0; i < max_tasks; ++i) {
                    const auto &r = records[i];
                    if (r.end <= r.begin) continue;
                    const bool gate = i < slots * ff / gate_width;
                    const int task = gate ? i : i - slots * ff / gate_width;
                    const int bytes = gate ? 2 * s.first_bytes / ff * gate_width : s.down_bytes / hidden * down_width;
                    std::printf("%d,%d,%d,%d,%d,%d,%d,%s,%d,%u,%d,%llu,%llu\n",
                        frame.batch, frame.layer, frame.cohort, frame.step, p, trial, mode,
                        gate ? "gate" : "down", task, r.routes, bytes, r.begin, r.end);
                }
            }
        }
    }
    checked(cudaFree(stamps));
}
