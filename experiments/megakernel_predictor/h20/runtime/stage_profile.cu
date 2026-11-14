// Reuse the compiled operators without needing the llama CUDA backend.
#include "resident-moe.cu"
#include "gguf.h"

void ggml_cuda_error(const char *message, const char *, const char *, int, const char *) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
}

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> T *allocate(size_t count) {
    T *p = nullptr;
    checked(cudaMalloc(&p, count * sizeof(T)));
    return p;
}

struct Data {
    int layer, batch, cohort, step;
    std::vector<float> activation, logits, output;
    std::vector<int> ids;
};

std::vector<Data> read_frames(const char *file, int limit) {
    std::ifstream input(file, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open frame data");
    std::vector<Data> frames;
    std::array<int, 40> count{};
    while (input.peek() != EOF) {
        int header[4]{};
        input.read(reinterpret_cast<char *>(header), sizeof(header));
        const int layer = header[0], batch = header[1];
        if (!input || layer < 0 || layer >= 40 || batch < 1 || batch > 8)
            throw std::runtime_error("invalid decode capture");
        Data frame{layer, batch, header[2], header[3], {}, {}, {}, {}};
        auto next = [&](auto &v, int n) {
            v.resize(n);
            input.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(v.front()));
        };
        next(frame.activation, batch * hidden);
        next(frame.logits, batch * n_experts);
        next(frame.ids, batch * top);
        next(frame.output, batch * hidden);
        if (!input) throw std::runtime_error("truncated decode capture");
        if (count[layer]++ < limit) frames.push_back(std::move(frame));
    }
    std::stable_sort(frames.begin(), frames.end(), [](const auto &a, const auto &b) { return a.layer < b.layer; });
    return frames;
}

struct Matrix {
    Weight *device = nullptr;
    int type = 0, bytes = 0;
};

class Checkpoint {
    std::ifstream input;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta;
public:
    Checkpoint(const char *path) : input(path, std::ios::binary),
        meta(gguf_init_from_file(path, {true, nullptr}), gguf_free) {
        if (!input || !meta) throw std::runtime_error("cannot read GGUF");
    }
    Matrix read(int layer, const char *name) {
        const auto key = "blk." + std::to_string(layer) + ".ffn_" + name + "_exps.weight";
        const auto index = gguf_find_tensor(meta.get(), key.c_str());
        if (index < 0) throw std::runtime_error("missing " + key);
        const size_t size = gguf_get_tensor_size(meta.get(), index);
        std::vector<Weight> host(size);
        input.seekg(gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), index));
        if (!input.read(reinterpret_cast<char *>(host.data()), size)) throw std::runtime_error("truncated " + key);
        auto *p = allocate<Weight>(size);
        checked(cudaMemcpy(p, host.data(), size, cudaMemcpyHostToDevice));
        return {p, gguf_get_tensor_type(meta.get(), index), static_cast<int>(size / n_experts)};
    }
};

struct Timings {
    // Per replay captures for independent service times.
    cudaStream_t stream{};
    cudaEvent_t start{}, end{};
    Timings() {
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checked(cudaEventCreate(&start));
        checked(cudaEventCreate(&end));
    }
    template<class Launch> float measure(Launch fn) {
        cudaGraph_t graph{};
        cudaGraphExec_t exec{};
        checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        fn();
        checked(cudaStreamEndCapture(stream, &graph));
        checked(cudaGraphInstantiate(&exec, graph, 0));
        checked(cudaGraphLaunch(exec, stream));
        constexpr int repetitions = 12;
        checked(cudaEventRecord(start, stream));
        for (int i = 0; i < repetitions; ++i) checked(cudaGraphLaunch(exec, stream));
        checked(cudaEventRecord(end, stream));
        checked(cudaEventSynchronize(end));
        float ms = 0;
        checked(cudaEventElapsedTime(&ms, start, end));
        checked(cudaGraphExecDestroy(exec));
        checked(cudaGraphDestroy(graph));
        return ms * 1000 / repetitions;
    }
};

template<bool cooperative, bool barrier>
__global__ void overhead_kernel() {
    if constexpr (cooperative && barrier) cg::this_grid().sync();
}

template<bool cooperative, bool barrier>
void overhead(cudaStream_t stream, int nsm) {
    if constexpr (cooperative) {
        checked(cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(overhead_kernel<cooperative, barrier>),
                                             nsm, threads, nullptr, 0, stream));
    } else {
        overhead_kernel<false, false><<<nsm, threads, 0, stream>>>();
        checked(cudaGetLastError());
    }
}

template<int batch>
void measure(State &s, Timings &t, int nsm, const Data &frame) {
    launch<batch, ROUTE>(t.stream, nsm, s);
    launch<batch, XQ>(t.stream, nsm, s);
    launch<batch, FIRST>(t.stream, nsm, s);
    launch<batch, HQ>(t.stream, nsm, s);
    launch<batch, DOWN>(t.stream, nsm, s);
    checked(cudaStreamSynchronize(t.stream));
    Routing route_result{};
    checked(cudaMemcpy(&route_result, s.routes, sizeof(route_result), cudaMemcpyDeviceToHost));
    if (!std::equal(frame.ids.begin(), frame.ids.end(), route_result.ids))
        throw std::runtime_error("captured expert IDs changed");
    const float route_us = t.measure([&] { launch<batch, ROUTE>(t.stream, nsm, s); });
    const float xq_us = t.measure([&] { launch<batch, XQ>(t.stream, nsm, s); });
    const float hq_us = t.measure([&] { launch<batch, HQ>(t.stream, nsm, s); });
    const float first_us = t.measure([&] { launch<batch, FIRST>(t.stream, nsm, s); });
    const float down_us = t.measure([&] { launch<batch, DOWN>(t.stream, nsm, s); });
    const float reduce_us = t.measure([&] { launch<batch, REDUCE>(t.stream, nsm, s); });
    const float empty = t.measure([&] { overhead<false, false>(t.stream, nsm); });
    const float coop = t.measure([&] { overhead<true, false>(t.stream, nsm); });
    const float barrier = t.measure([&] { overhead<true, true>(t.stream, nsm); });
    std::printf("%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                frame.batch, frame.layer, frame.cohort, frame.step, route_result.used,
                route_us, xq_us, first_us, hq_us, down_us, reduce_us, empty, coop, barrier);
}
} // namespace

int main(int argc, char **argv) try {
    if (argc != 4) throw std::runtime_error("usage: stage_profile MODEL CAPTURE FORWARD_LIMIT");
    const auto frames = read_frames(argv[2], std::stoi(argv[3]));
    if (frames.empty()) throw std::runtime_error("empty capture");
    Checkpoint checkpoint(argv[1]);
    cudaDeviceProp prop{};
    checked(cudaGetDeviceProperties(&prop, 0));
    Timings t;
    auto *activation = allocate<float>(max_batch * hidden);
    auto *logits = allocate<float>(max_batch * n_experts);
    auto *ids = allocate<int>(slots);
    State s{};
    s.activation = activation;
    s.logits = logits;
    s.fixed_ids = ids;
    s.logit_stride = n_experts;
    s.id_stride = top;
    s.intermediate = allocate<float>(slots * ff);
    s.down_output = allocate<float>(slots * hidden);
    s.out = allocate<float>(max_batch * hidden);
    s.routes = allocate<Routing>(1);
    s.xq = allocate<block_q8_1>(max_batch * hidden / warp);
    s.hq = allocate<block_q8_1>(slots * ff / warp);
    s.score_scale = 1;
    std::array<Matrix, 3> weights{};
    int previous = -1;
    std::puts("batch,layer,cohort,step,unique,route_us,xq_us,first_us,hq_us,down_us,reduce_us,empty_us,coop_us,barrier_us");
    for (const auto &frame : frames) {
        if (frame.layer != previous) {
            for (auto &matrix : weights) if (matrix.device) checked(cudaFree(matrix.device));
            weights = {checkpoint.read(frame.layer, "gate"), checkpoint.read(frame.layer, "up"),
                       checkpoint.read(frame.layer, "down")};
            s.gate = weights[0].device;
            s.up = weights[1].device;
            s.down = weights[2].device;
            s.first_type = weights[0].type;
            s.first_bytes = weights[0].bytes;
            s.down_type = weights[2].type;
            s.down_bytes = weights[2].bytes;
            previous = frame.layer;
        }
        s.batch = frame.batch;
        checked(cudaMemcpy(activation, frame.activation.data(), frame.activation.size() * sizeof(float), cudaMemcpyHostToDevice));
        checked(cudaMemcpy(logits, frame.logits.data(), frame.logits.size() * sizeof(float), cudaMemcpyHostToDevice));
        checked(cudaMemcpy(ids, frame.ids.data(), frame.ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (frame.batch <= 2) measure<2>(s, t, prop.multiProcessorCount, frame);
        else if (frame.batch <= 4) measure<4>(s, t, prop.multiProcessorCount, frame);
        else measure<8>(s, t, prop.multiProcessorCount, frame);
    }
} catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
