#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

size_t used() {
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    return total - free;
}

void add(llama_batch &batch, llama_token token, int position, int seq, bool logits) {
    int i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = position;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = seq;
    batch.logits[i] = logits;
}

std::vector<std::vector<llama_token>> prompts(const llama_vocab *vocab, const char *path) {
    std::ifstream file(path);
    std::vector<std::vector<llama_token>> list;
    for (std::string text; std::getline(file, text); ) {
        if (text.empty()) continue;
        std::vector<llama_token> ids(text.size() + 16);
        int n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), true, false);
        if (n < 16 || n > 512) throw std::runtime_error("invalid test prompt size");
        ids.resize(n);
        list.emplace_back(std::move(ids));
    }
    return list;
}

int main(int argc, char **argv) try {
    if (argc != 7) throw std::runtime_error("usage: bench MODEL PROMPTS BATCH REQUESTS DECODE_STEPS RECORD_PATH");
    const int concurrency = std::stoi(argv[3]), requests = std::stoi(argv[4]), steps = std::stoi(argv[5]);
    if (concurrency < 1 || concurrency > 8 || requests < concurrency || requests % concurrency || steps < 1)
        throw std::runtime_error("invalid batch, requests or steps");
    llama_backend_init();
    const auto gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu) throw std::runtime_error("CUDA device unavailable");
    const auto model_memory_start = used();
    const llama_model_tensor_buft_override overrides[] = {
        {"^token_embd\\.weight$", ggml_backend_dev_buffer_type(gpu)},
        {nullptr, nullptr},
    };
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts = false;
    mp.no_host = true;
    mp.tensor_buft_overrides = overrides;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    const auto loaded = used();
    const auto *vocab = llama_model_get_vocab(model.get());
    const int vocabulary = llama_vocab_n_tokens(vocab);
    const auto data = prompts(vocab, argv[2]);
    if ((int)data.size() < requests) throw std::runtime_error("too few prompts");

    auto cp = llama_context_default_params();
    cp.n_ctx = 640 * concurrency;
    cp.n_seq_max = concurrency;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.offload_kqv = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("context init failed");
    const auto memory = llama_get_memory(ctx.get());
    llama_batch batch = llama_batch_init(cp.n_batch, 0, 1);

    std::vector<llama_token> teacher;
    std::ifstream input_teacher(argv[6], std::ios::binary);
    if (input_teacher) {
        teacher.resize(requests * steps);
        if (!input_teacher.read(reinterpret_cast<char *>(teacher.data()), teacher.size() * sizeof(llama_token)) ||
            input_teacher.peek() != EOF)
            throw std::runtime_error("wrong teacher token count");
    }
    std::ofstream record;
    if (teacher.empty()) {
        record.open(argv[6], std::ios::binary);
        if (!record) throw std::runtime_error("cannot create teacher file");
    }
    const std::string prediction = std::string(argv[6]) + ".reference";
    const bool reference_mode = teacher.empty() || std::getenv("QWEN_RECORD_REFERENCE");
    std::ifstream reference;
    std::ofstream reference_out;
    if (!std::getenv("QWEN_SKIP_REFERENCE")) {
        if (reference_mode) reference_out.open(prediction, std::ios::binary);
        else reference.open(prediction, std::ios::binary);
        if (!reference.is_open() && !reference_out.is_open()) throw std::runtime_error("cannot open reference logits");
    }

    auto run = [&] {
        const auto start = Clock::now();
        const int status = llama_decode(ctx.get(), batch);
        llama_synchronize(ctx.get());
        if (status) throw std::runtime_error("llama_decode failed");
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };
    auto argmax = [&](const float *logits) {
        return llama_token(std::max_element(logits, logits + vocabulary) - logits);
    };
    auto mark = reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT, "qwen_trace_mark"));
    int matches = 0, logits_seen = 0;
    double max_l2 = 0;
    std::vector<float> expected(vocabulary);
    printf("batch,cohort,step,decode_ms,peak_bytes\n");
    size_t peak = loaded;
    for (int start = 0; start < requests; start += concurrency) {
        llama_memory_clear(memory, true);
        std::vector<llama_token> next(concurrency);
        for (int row = 0; row < concurrency; ++row) {
            const auto &ids = data[start + row];
            batch.n_tokens = 0;
            for (size_t pos = 0; pos < ids.size(); ++pos) add(batch, ids[pos], pos, row, pos + 1 == ids.size());
            run();
            next[row] = argmax(llama_get_logits_ith(ctx.get(), -1));
        }
        for (int step = 0; step < steps; ++step) {
            batch.n_tokens = 0;
            for (int row = 0; row < concurrency; ++row) {
                const auto token = teacher.empty() ? next[row] : teacher[start * steps + step * concurrency + row];
                if (teacher.empty()) {
                    // Record in step-major order within a concurrent cohort.
                    record.write(reinterpret_cast<const char *>(&token), sizeof(token));
                }
                add(batch, token,
                    data[start + row].size() + step, row, true);
            }
            if (mark) mark(1);
            const auto elapsed = run();
            if (mark) mark(0);
            peak = std::max(peak, used());
            printf("%d,%d,%d,%.6f,%zu\n", concurrency, start / concurrency, step, elapsed, peak);
            for (int row = 0; row < concurrency; ++row) {
                const float *logits = llama_get_logits_ith(ctx.get(), row);
                next[row] = argmax(logits);
                if (!reference.is_open() && !reference_out.is_open()) continue;
                if (reference_out.is_open()) {
                    reference_out.write(reinterpret_cast<const char *>(logits), vocabulary * sizeof(float));
                } else {
                    if (!reference.read(reinterpret_cast<char *>(expected.data()), vocabulary * sizeof(float)))
                        throw std::runtime_error("truncated reference logits");
                    double sq = 0, norm = 0;
                    for (int i = 0; i < vocabulary; ++i) {
                        const double v = logits[i], ref = expected[i];
                        if (!std::isfinite(v) || !std::isfinite(ref)) throw std::runtime_error("non-finite logits");
                        sq += (v - ref) * (v - ref);
                        norm += ref * ref;
                    }
                    max_l2 = std::max(max_l2, std::sqrt(sq / std::max(norm, 1e-30)));
                    matches += next[row] == argmax(expected.data());
                    ++logits_seen;
                }
            }
        }
        fflush(stdout);
    }
    fprintf(stderr, "device_model_bytes=%zu peak_bytes=%zu argmax_matches=%d/%d max_logit_l2=%.8f\n",
            loaded - model_memory_start, peak, matches, logits_seen, max_l2);
    llama_batch_free(batch);
    return 0;
} catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
}
