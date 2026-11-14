// Each block holds one expert's intermediates in shared memory. A ready block
// proceeds to down without synchronizing with other experts.
template<int width>
__device__ __forceinline__ void quantize_expert(const float *in, block_q8_1 *out, int nrows) {
    const int lane = threadIdx.x % warp;
    for (int i = threadIdx.x / warp; i < nrows * (width / warp); i += blockDim.x / warp) {
        const float x = in[i * warp + lane];
        const float max = warp_reduce_max<warp>(fabsf(x));
        const float sum = warp_reduce_sum<warp>(x);
        const float scale = max / 127.f;
        out[i].qs[lane] = max == 0 ? 0 : roundf(x / scale);
        if (!lane) out[i].ds = make_half2(scale, sum);
    }
}

template<bool first, int type>
__device__ __forceinline__ void project_expert(State s, const Group &group, float *h, const block_q8_1 *hq) {
    const int lane = threadIdx.x % warp;
    constexpr int cols = first ? hidden : ff;
    constexpr int rows = first ? ff : hidden;
    const int row_bytes = first ? s.first_bytes / ff : s.down_bytes / hidden;
    for (int row = threadIdx.x / warp; row < rows; row += blockDim.x / warp) {
        const int expert = group.expert;
        const auto *a = first ? s.gate + expert * s.first_bytes + row * row_bytes
                              : s.down + expert * s.down_bytes + row * row_bytes;
        const auto *b = s.up + expert * s.first_bytes + row * row_bytes;
        for (int j = 0; j < group.count; ++j) {
            const int idx = group.routes[j];
            const auto *x = first ? s.xq + idx / top * (hidden / warp) : hq + j * (ff / warp);
            float ga = 0, up = 0;
            for (int col = lane; col < cols / warp; col += warp) {
                ga += dot<type>(a, x, col);
                if constexpr (first) up += dot<type>(b, x, col);
            }
            ga = sum_warp(ga);
            if constexpr (first) up = sum_warp(up);
            if (lane == 0) {
                if constexpr (first) h[j * ff + row] = up * (ga / (1.0f + expf(-ga)));
                else s.down_output[idx * hidden + row] = ga;
            }
        }
    }
}

template<int batch, bool wait_all>
__global__ void expert_local(State s) {
    extern __shared__ float scratch[];
    float *h = scratch;
    auto *hq = reinterpret_cast<block_q8_1 *>(h + batch * ff);
    auto grid = cg::this_grid();
    route(s);
    quantize<hidden>(s.activation, s.xq, s.batch);
    grid.sync();
    const int i = blockIdx.x;
    if (i < s.routes->used) {
        const auto &group = s.routes->groups[i];
        if (s.first_type == iq2) project_expert<true, iq2>(s, group, h, hq);
        else if (s.first_type == iq3) project_expert<true, iq3>(s, group, h, hq);
        else project_expert<true, iq4>(s, group, h, hq);
        __syncthreads();
    }
    if constexpr (wait_all) grid.sync();
    if (i < s.routes->used) {
        quantize_expert<ff>(h, hq, s.routes->groups[i].count);
        __syncthreads();
    }
    if constexpr (wait_all) grid.sync();
    if (i < s.routes->used) {
        const auto &group = s.routes->groups[i];
        if (s.down_type == iq2) project_expert<false, iq2>(s, group, h, hq);
        else if (s.down_type == iq3) project_expert<false, iq3>(s, group, h, hq);
        else project_expert<false, iq4>(s, group, h, hq);
    }
    grid.sync();
    reduce(s);
}
