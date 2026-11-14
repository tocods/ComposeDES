#include "resident-moe.cuh"
#include "vecdotq.cuh"
#include <cooperative_groups.h>
#include <math_constants.h>
#include <cstring>

namespace {
namespace cg = cooperative_groups;
constexpr int hidden = 2048, ff = 512, n_experts = 256, top = 8, max_batch = 8;
constexpr int slots = max_batch * top, threads = 256, warp = 32, warps = threads / warp;
constexpr int iq2 = GGML_TYPE_IQ2_XS, iq3 = GGML_TYPE_IQ3_XXS, iq4 = GGML_TYPE_IQ4_XS;

struct Group {
    int expert, count;
    int routes[max_batch];
};
struct Routing {
    Group groups[slots];
    int used;
    int ids[slots];
    float score[slots];
};
using Weight = unsigned char;

struct State {
    const float *activation, *logits;
    const int32_t *fixed_ids;
    size_t logit_stride, id_stride;
    float *intermediate, *down_output, *out;
    block_q8_1 *xq, *hq;
    const Weight *gate, *up, *down;
    Routing *routes;
    int batch, first_type, first_bytes, down_type, down_bytes;
    float score_scale;
};

__device__ float sum_warp(float x) {
    for (int i = warp / 2; i; i >>= 1) x += __shfl_down_sync(0xffffffff, x, i);
    return x;
}

__device__ __forceinline__ void route(State s) {
    if (blockIdx.x != 0) return;
    const int w = threadIdx.x / warp, lane = threadIdx.x % warp;
    Routing &r = *s.routes;
    if (w < s.batch) {
        const float *logits = s.logits + w * s.logit_stride;
        int id = -1;
        const auto *fixed = s.fixed_ids ? s.fixed_ids + w * s.id_stride : nullptr;
        float values[n_experts / warp];
#pragma unroll
        for (int i = 0; i < n_experts / warp; ++i) values[i] = logits[i * warp + lane];
        float weights[top], normalizer = 0, largest = 0;
        for (int k = 0; k < top; ++k) {
            float best = -CUDART_INF_F;
            if (fixed) {
                id = fixed[k];
                best = logits[id];
            } else {
                id = n_experts;
#pragma unroll
                for (int i = 0; i < n_experts / warp; ++i) {
                    if (values[i] > best) {
                        best = values[i];
                        id = i * warp + lane;
                    }
                }
                for (int stride = warp / 2; stride; stride >>= 1) {
                    float peer = __shfl_down_sync(0xffffffff, best, stride);
                    int other = __shfl_down_sync(0xffffffff, id, stride);
                    if (peer > best || (peer == best && other < id)) {
                        best = peer;
                        id = other;
                    }
                }
                id = __shfl_sync(0xffffffff, id, 0);
                if (lane == (id % warp)) values[id / warp] = -CUDART_INF_F;
            }
            if (lane == 0) {
                if (k == 0) largest = best;
                weights[k] = expf(best - largest);
                normalizer += weights[k];
                r.ids[w * top + k] = id;
            }
        }
        if (lane == 0)
            for (int k = 0; k < top; ++k)
                r.score[w * top + k] = s.score_scale * weights[k] / normalizer;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int used = 0;
        // Group tokens by expert so projections reuse a quantized weight on each row.
        __shared__ int index[n_experts];
        for (int e = 0; e < n_experts; ++e) index[e] = -1;
        for (int i = 0; i < s.batch * top; ++i) {
            const int expert = r.ids[i];
            int group = index[expert];
            if (group < 0) {
                index[expert] = group = used++;
                r.groups[group].expert = expert;
                r.groups[group].count = 0;
            }
            auto &target = r.groups[group];
            target.routes[target.count++] = i;
        }
        r.used = used;
    }
}

template<int width>
__device__ __forceinline__ void quantize(const float *input, block_q8_1 *output, int nrows) {
    const int lane = threadIdx.x % warp;
    const int worker = (blockIdx.x * blockDim.x + threadIdx.x) / warp;
    const int workers = gridDim.x * blockDim.x / warp;
    for (int i = worker; i < nrows * (width / warp); i += workers) {
        const float x = input[i * warp + lane];
        const float max = warp_reduce_max<warp>(fabsf(x));
        const float sum = warp_reduce_sum<warp>(x);
        const float scale = max / 127.f;
        output[i].qs[lane] = max == 0 ? 0 : roundf(x / scale);
        if (!lane) output[i].ds = make_half2(scale, sum);
    }
}

template<int type>
__device__ __forceinline__ float dot(const Weight *w, const block_q8_1 *x, int block32) {
    const int row_block = block32 / 8;
    const auto *q8 = x + row_block * 8;
    if constexpr (type == iq2) return vec_dot_iq2_xs_q8_1(w, q8, row_block, 2 * (block32 % 8));
    if constexpr (type == iq3) return vec_dot_iq3_xxs_q8_1(w, q8, row_block, 2 * (block32 % 8));
    if constexpr (type == iq4) return vec_dot_iq4_xs_q8_1(w, q8, row_block, 4 * (block32 % 8));
}

template<bool first, int batch, int type>
__device__ __forceinline__ void project_quant(State s) {
    const int lane = threadIdx.x % warp;
    const int worker = int(blockIdx.x) * warps + threadIdx.x / warp;
    const int workers = gridDim.x * warps;
    constexpr int cols = first ? hidden : ff;
    constexpr int rows = first ? ff : hidden;
    const auto &r = *s.routes;
    const int row_bytes = first ? s.first_bytes / ff : s.down_bytes / hidden;
    for (int i = worker; i < r.used * rows; i += workers) {
        const int g = i / rows, row = i % rows;
        const Group &group = r.groups[g];
        const int expert = group.expert;
        const auto *a = first ? s.gate + expert * s.first_bytes + row * row_bytes
                              : s.down + expert * s.down_bytes + row * row_bytes;
        const auto *b = s.up + expert * s.first_bytes + row * row_bytes;
        float sum_a[batch] = {}, sum_b[batch] = {};
        for (int col = lane; col < cols / warp; col += warp) {
#pragma unroll
            for (int j = 0; j < batch; ++j) {
                if (j < group.count) {
                    const int idx = group.routes[j];
                    const auto *x = first ? s.xq + idx / top * (hidden / warp)
                                          : s.hq + idx * (ff / warp);
                    sum_a[j] += dot<type>(a, x, col);
                    if constexpr (first) sum_b[j] += dot<type>(b, x, col);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < batch; ++j) {
            if (j < group.count) {
                const float ga = sum_warp(sum_a[j]);
                float up = 0;
                if constexpr (first) up = sum_warp(sum_b[j]);
                if (lane == 0) {
                    const int idx = group.routes[j];
                    if constexpr (first) s.intermediate[idx * ff + row] = up * (ga / (1.0f + expf(-ga)));
                    else s.down_output[idx * hidden + row] = ga;
                }
            }
        }
    }
}

template<bool first, int batch>
__device__ __forceinline__ void project(State s) {
    switch (first ? s.first_type : s.down_type) {
        case iq2: project_quant<first, batch, iq2>(s); break;
        case iq3: project_quant<first, batch, iq3>(s); break;
        case iq4: project_quant<first, batch, iq4>(s); break;
    }
}

__device__ __forceinline__ void reduce(State s) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < s.batch * hidden; i += gridDim.x * blockDim.x) {
        const int token = i / hidden, column = i % hidden;
        float total = 0;
        for (int k = 0; k < top; ++k) {
            const int id = token * top + k;
            total += s.routes->score[id] * s.down_output[id * hidden + column];
        }
        s.out[i] = total;
    }
}

enum Stage : unsigned { ROUTE = 1, FIRST = 2, DOWN = 4, REDUCE = 8, XQ = 16, HQ = 32 };
enum Plan { WHOLE = 1, ROUTE_CUT, GATE_CUT, FOUR };
template<int batch, unsigned mask>
__global__ void resident_kernel(State s) {
    auto grid = cg::this_grid();
    if constexpr (mask & ROUTE) route(s);
    if constexpr (mask & XQ) quantize<hidden>(s.activation, s.xq, s.batch);
    if constexpr ((mask & (ROUTE | XQ)) && (mask & FIRST)) grid.sync();
    if constexpr (mask & FIRST) project<true, batch>(s);
    if constexpr ((mask & FIRST) && (mask & (HQ | DOWN | REDUCE))) grid.sync();
    if constexpr (mask & HQ) quantize<ff>(s.intermediate, s.hq, s.batch * top);
    if constexpr ((mask & HQ) && (mask & DOWN)) grid.sync();
    if constexpr (mask & DOWN) project<false, batch>(s);
    if constexpr ((mask & DOWN) && (mask & REDUCE)) grid.sync();
    if constexpr (mask & REDUCE) reduce(s);
}

template<int batch, unsigned stages>
void launch(cudaStream_t stream, int blocks, State &s) {
    constexpr bool multi = stages & (stages - 1);
    if constexpr (multi) {
        void *args[] = {&s};
        CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(resident_kernel<batch, stages>),
                                               blocks, threads, args, 0, stream));
    } else {
        const int grid = stages == ROUTE ? 1 : blocks;
        resident_kernel<batch, stages><<<grid, threads, 0, stream>>>(s);
        CUDA_CHECK(cudaGetLastError());
    }
}

template<int batch>
void dispatch(cudaStream_t stream, int blocks, int mode, State &s) {
    switch (mode) {
        case WHOLE:
            launch<batch, ROUTE | XQ | FIRST | HQ | DOWN | REDUCE>(stream, blocks, s);
            break;
        case ROUTE_CUT:
            launch<batch, ROUTE>(stream, blocks, s);
            launch<batch, XQ | FIRST | HQ | DOWN | REDUCE>(stream, blocks, s);
            break;
        case GATE_CUT:
            launch<batch, ROUTE | XQ | FIRST>(stream, blocks, s);
            launch<batch, HQ | DOWN | REDUCE>(stream, blocks, s);
            break;
        case FOUR:
            launch<batch, ROUTE>(stream, blocks, s);
            launch<batch, XQ>(stream, blocks, s);
            launch<batch, FIRST>(stream, blocks, s);
            launch<batch, HQ>(stream, blocks, s);
            launch<batch, DOWN>(stream, blocks, s);
            launch<batch, REDUCE>(stream, blocks, s);
            break;
        default:
            GGML_ABORT("invalid resident MoE cut");
    }
}

bool quant_supported(int q) {
    return q == iq2 || q == iq3 || q == iq4;
}
} // namespace

bool ggml_cuda_is_resident_moe(const ggml_tensor *op) {
    return op->op == GGML_OP_CUSTOM && !std::strncmp(op->name, "qwen_resident_moe", 17);
}

#ifndef QWEN_RESIDENT_STANDALONE
void ggml_cuda_resident_moe(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    const auto *activation = dst->src[0], *logits = dst->src[1], *gate = dst->src[2];
    const auto *up = dst->src[3], *down = dst->src[4];
    const auto *fixed = dst->src[5];
    GGML_ASSERT(activation->type == GGML_TYPE_F32 && logits->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(activation) && ggml_is_contiguous(gate) &&
                ggml_is_contiguous(up) && ggml_is_contiguous(down));
    GGML_ASSERT(gate->type == up->type && quant_supported(gate->type) && quant_supported(down->type));
    GGML_ASSERT(!fixed || (fixed->type == GGML_TYPE_I32 && fixed->nb[0] == sizeof(int32_t)));
    const int batch = activation->ne[1];
    GGML_ASSERT(batch >= 1 && batch <= max_batch && logits->ne[1] == batch);
    const int mode = dst->name[17] - '0';
    float scale = 1.f;
    memcpy(&scale, &dst->op_params[12], sizeof(scale));
    const size_t scratch_bytes = sizeof(Routing) + sizeof(float) * batch * top * (ff + hidden) +
        sizeof(block_q8_1) * batch * (hidden + top * ff) / warp;
    GGML_ASSERT(ggml_nbytes(dst) >= batch * hidden * sizeof(float) + scratch_bytes);
    // Scratch and output share one graph-planned allocation. Intermediate pointers
    // therefore remain valid across CUDA Graph recording and all following replays.
    auto *out = static_cast<float *>(dst->data);
    auto *route_state = reinterpret_cast<Routing *>(out + batch * hidden);
    auto *intermediate = reinterpret_cast<float *>(route_state + 1);
    auto *xq = reinterpret_cast<block_q8_1 *>(intermediate + batch * top * (ff + hidden));
    State s{
        static_cast<const float *>(activation->data), static_cast<const float *>(logits->data),
        fixed ? static_cast<const int32_t *>(fixed->data) : nullptr,
        logits->nb[1] / sizeof(float), fixed ? fixed->nb[1] / sizeof(int32_t) : 0,
        intermediate, intermediate + batch * top * ff, out, xq, xq + batch * hidden / warp,
        static_cast<const Weight *>(gate->data), static_cast<const Weight *>(up->data),
        static_cast<const Weight *>(down->data), route_state,
        batch, gate->type, int(gate->nb[2]), down->type, int(down->nb[2]), scale,
    };
    const int blocks = ggml_cuda_info().devices[ctx.device].nsm;
    const auto stream = ctx.stream();
    if (batch <= 2) dispatch<2>(stream, blocks, mode, s);
    else if (batch <= 4) dispatch<4>(stream, blocks, mode, s);
    else dispatch<8>(stream, blocks, mode, s);
}
#endif
