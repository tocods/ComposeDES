#include "llama.h"
#include "ggml-backend.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <type_traits>
#include <string_view>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;

struct Request {
    std::vector<llama_token> prompt;
    std::vector<llama_token> generated;
    clock_type::time_point arrival;
    double ttft_ms = 0;
    double tail_ms = 0;
};

struct Slot {
    int request = -1;
    int position = 0;
};

static double milliseconds(clock_type::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

static size_t cuda_used() {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
        throw std::runtime_error("cudaMemGetInfo failed");
    }
    return total - free;
}

static void add(llama_batch & batch, llama_token token, int position, int seq, bool logits) {
    const int at = batch.n_tokens++;
    batch.token[at] = token;
    batch.pos[at] = position;
    batch.n_seq_id[at] = 1;
    batch.seq_id[at][0] = seq;
    batch.logits[at] = logits;
}

static std::vector<Request> read_prompts(const llama_vocab * vocab, const char * path, int count) {
    std::ifstream input(path);
    std::vector<Request> requests;
    for (std::string prompt; int(requests.size()) < count && std::getline(input, prompt); ) {
        if (prompt.empty()) {
            continue;
        }
        Request request;
        request.prompt.resize(prompt.size() + 16);
        const int n = llama_tokenize(vocab, prompt.data(), prompt.size(), request.prompt.data(),
                                     request.prompt.size(), true, false);
        if (n < 17 || n > 512) {
            throw std::runtime_error("expected prompts of 17 to 512 tokens");
        }
        request.prompt.resize(n);
        requests.push_back(std::move(request));
    }
    if (int(requests.size()) != count) {
        throw std::runtime_error("too few prompts");
    }
    return requests;
}

// Each layer record contains the actual MoE input, router logits and expert IDs.
struct Frame {
    std::vector<float> activation, logits, output;
    std::vector<int32_t> ids;
};
struct Capture {
    int batch, cohort = 0, step = 0;
    bool enabled = false, failed = false;
    std::map<int, Frame> layers;
};
static bool capture_moe(ggml_tensor *t, bool ask, void *data) {
    auto &capture = *static_cast<Capture *>(data);
    if (!capture.enabled) return !ask;
    std::string_view name(t->name);
    const std::array<std::string_view, 4> prefixes = {
        "attn_post_norm-", "ffn_moe_logits-", "ffn_moe_topk-", "ffn_moe_out-"
    };
    for (int stage = 0; stage < 4; ++stage) {
        auto prefix = prefixes[stage];
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        auto suffix = name.substr(prefix.size());
        if (suffix.empty() || suffix.find_first_not_of("0123456789") != std::string_view::npos)
            return !ask;
        if (ask) return true;
        int layer = std::stoi(std::string(suffix));
        auto &frame = capture.layers[layer];
        const int width[] = {2048, 256, 8, 2048};
        if (t->ne[0] < width[stage] || t->ne[1] != capture.batch || t->nb[0] != sizeof(float)
                || t->type != (stage == 2 ? GGML_TYPE_I32 : GGML_TYPE_F32)) {
            capture.failed = true;
            fprintf(stderr, "unexpected capture tensor %s (%lld,%lld,%lld) type %d\n",
                    t->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], t->type);
            return false;
        }
        if (stage == 2) frame.ids.resize(capture.batch * width[stage]);
        auto *values = stage == 0 ? &frame.activation : stage == 1 ? &frame.logits : &frame.output;
        if (stage != 2) values->resize(capture.batch * width[stage]);
        std::vector<char> bytes(ggml_nbytes(t));
        ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        char *destination = stage == 2 ? reinterpret_cast<char *>(frame.ids.data())
                                       : reinterpret_cast<char *>(values->data());
        for (int row = 0; row < capture.batch; ++row)
            std::memcpy(destination + row * width[stage] * sizeof(float),
                        bytes.data() + row * t->nb[1], width[stage] * sizeof(float));
        return true;
    }
    return !ask;
}
static void save_capture(std::ofstream &file, const Capture &capture) {
    if (capture.failed || capture.layers.size() != 40) {
        for (const auto &[layer, frame] : capture.layers)
            fprintf(stderr, "capture %d: %zu %zu %zu %zu\n", layer, frame.activation.size(),
                    frame.logits.size(), frame.ids.size(), frame.output.size());
        throw std::runtime_error("incomplete forward capture: " + std::to_string(capture.layers.size()));
    }
    auto write = [&](const auto &values, size_t expected) {
        if (values.size() != expected) throw std::runtime_error("incomplete MoE tensor");
        using Value = typename std::decay_t<decltype(values)>::value_type;
        file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(Value));
    };
    for (const auto &[layer, frame] : capture.layers) {
        const std::array<int, 4> header{layer, capture.batch, capture.cohort, capture.step};
        file.write(reinterpret_cast<const char *>(header.data()), sizeof(header));
        write(frame.activation, 2048 * capture.batch);
        write(frame.logits, 256 * capture.batch);
        write(frame.ids, 8 * capture.batch);
        write(frame.output, 2048 * capture.batch);
    }
    if (!file) throw std::runtime_error("could not write forward capture");
}

int main(int argc, char ** argv) try {
    if (argc != 8) {
        fprintf(stderr, "usage: %s model prompts concurrency requests output_tokens budget_MiB CAPTURE_BIN\n", argv[0]);
        return 2;
    }
    const int concurrency = std::stoi(argv[3]);
    const int count = std::stoi(argv[4]);
    const int output_tokens = std::stoi(argv[5]);
    const int variation = std::getenv("QWEN_BENCH_OUTPUT_SPREAD") ? std::stoi(std::getenv("QWEN_BENCH_OUTPUT_SPREAD")) : 0;
    const size_t budget = std::stoull(argv[6]) * 1024 * 1024;
    if (concurrency < 1 || concurrency > 16 || count < concurrency || output_tokens < 2 || output_tokens > 256 ||
        variation < 0 || variation >= output_tokens - 1 || output_tokens + variation > 256) {
        throw std::runtime_error("invalid concurrency, request count or output length");
    }
    std::ofstream capture_file(argv[7], std::ios::binary);
    if (!capture_file) throw std::runtime_error("could not open capture");
    Capture capture{concurrency};
    const bool record_every_step = std::getenv("QWEN_CAPTURE_EVERY_STEP") != nullptr;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts = false;
    mp.no_host = true;

    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) {
        throw std::runtime_error("model load failed");
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    const int vocab_size = llama_vocab_n_tokens(vocab);
    auto requests = read_prompts(vocab, argv[2], count);
    std::vector<int> target_tokens(count);
    for (int i = 0; i < count; ++i) {
        const int offset = (i % 3) - 1;
        target_tokens[i] = output_tokens + offset * variation;
        requests[i].generated.reserve(target_tokens[i]);
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024 * concurrency;
    cp.n_seq_max = concurrency;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.cb_eval = capture_moe;
    cp.cb_eval_user_data = &capture;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) {
        throw std::runtime_error("context allocation failed");
    }
    auto memory = llama_get_memory(ctx.get());
    llama_batch batch = llama_batch_init(cp.n_batch, 0, 1);
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
    size_t peak = cuda_used();
    const auto check_vram = [&] {
        peak = std::max(peak, cuda_used());
        if (budget && peak > budget) {
            throw std::runtime_error("CUDA memory cap exceeded");
        }
    };
    const auto evaluate = [&] {
        if (llama_decode(ctx.get(), batch) != 0) {
            throw std::runtime_error("llama_decode failed");
        }
        llama_synchronize(ctx.get());
    };
    const auto sample = [&](int row) {
        return llama_sampler_sample(sampler.get(), ctx.get(), row);
    };

    const auto & first = requests.front().prompt;
    for (int pos = 0; pos < 32; ++pos) {
        add(batch, first[pos % first.size()], pos, 0, pos == 31);
    }
    evaluate();
    const auto warm_token = llama_sampler_sample(sampler.get(), ctx.get(), 31);
    batch.n_tokens = 0;
    add(batch, warm_token, 32, 0, true);
    evaluate();
    check_vram();
    llama_memory_clear(memory, true);

    std::vector<Slot> slots(concurrency);
    int next = 0, finished = 0, decode_batches = 0, peak_batch = 0;
    double decode_wall_ms = 0;
    const auto test_start = clock_type::now();
    while (finished < count) {
        // Admit a replacement prompt into every vacant sequence. Prefill is
        // separate from the next decode batch and stalls already active requests.
        const auto arrival = clock_type::now();
        for (int seq = 0; seq < concurrency && next < count; ++seq) {
            auto & slot = slots[seq];
            if (slot.request != -1) {
                continue;
            }
            slot.request = next++;
            auto & request = requests[slot.request];
            request.arrival = arrival;
            batch.n_tokens = 0;
            for (size_t pos = 0; pos < request.prompt.size(); ++pos) {
                add(batch, request.prompt[pos], pos, seq, pos + 1 == request.prompt.size());
            }
            evaluate();
            request.generated.push_back(sample(batch.n_tokens - 1));
            request.ttft_ms = milliseconds(clock_type::now() - request.arrival);
            slot.position = request.prompt.size();
            check_vram();
        }
        batch.n_tokens = 0;
        std::vector<int> decoded;
        for (int seq = 0; seq < concurrency; ++seq) {
            auto & slot = slots[seq];
            if (slot.request >= 0) {
                add(batch, requests[slot.request].generated.back(), slot.position++, seq, true);
                decoded.push_back(seq);
            }
        }
        if (decoded.empty()) {
            throw std::runtime_error("benchmark stalled");
        }
        capture.step = decode_batches;
        capture.enabled = batch.n_tokens == concurrency &&
            (record_every_step || decode_batches % 23 == 4 || decode_batches % 23 == 15);
        capture.cohort = decode_batches / 23;
        ++decode_batches;
        peak_batch = std::max(peak_batch, batch.n_tokens);
        const auto decode_start = clock_type::now();
        evaluate();
        if (capture.enabled) {
            save_capture(capture_file, capture);
            capture.layers.clear();
        }
        capture.enabled = false;
        for (size_t row = 0; row < decoded.size(); ++row) {
            auto & slot = slots[decoded[row]];
            auto & request = requests[slot.request];
            request.generated.push_back(sample(row));
            if (int(request.generated.size()) == target_tokens[slot.request]) {
                request.tail_ms = milliseconds(clock_type::now() - request.arrival) - request.ttft_ms;
                if (!llama_memory_seq_rm(memory, decoded[row], -1, -1)) {
                    throw std::runtime_error("could not release sequence state");
                }
                slot.request = -1;
                ++finished;
            }
        }
        decode_wall_ms += milliseconds(clock_type::now() - decode_start);
        check_vram();
    }
    const double wall_ms = milliseconds(clock_type::now() - test_start);

    int total_output_tokens = 0;
    for (int i = 0; i < count; ++i) {
        const auto & request = requests[i];
        total_output_tokens += target_tokens[i];
        printf("{\"request\":%d,\"prompt_tokens\":%zu,\"ttft_ms\":%.6f,\"tpot_ms\":%.6f,\"tokens\":[",
               i, request.prompt.size(), request.ttft_ms, request.tail_ms / (target_tokens[i] - 1));
        for (size_t j = 0; j < request.generated.size(); ++j) {
            printf("%s%d", j ? "," : "", request.generated[j]);
        }
        puts("]}");
    }
    printf("{\"concurrency\":%d,\"requests\":%d,\"output_tokens\":%d,\"decode_batches\":%d,\"peak_batch\":%d,"
           "\"wall_ms\":%.6f,\"decode_wall_ms\":%.6f,\"throughput\":%.6f,\"decode_throughput\":%.6f,"
           "\"cuda_peak_bytes\":%zu}\n", concurrency, count, total_output_tokens, decode_batches, peak_batch,
           wall_ms, decode_wall_ms, total_output_tokens * 1000.0 / wall_ms,
           (total_output_tokens - count) * 1000.0 / decode_wall_ms, peak);
    llama_batch_free(batch);
    sampler.reset();
    ctx.reset();
    model.reset();
    llama_backend_free();
    return 0;
} catch (const std::exception & e) {
    fprintf(stderr, "benchmark error: %s\n", e.what());
    return 1;
}
