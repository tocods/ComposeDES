#include "persistent_kernel.cuh"
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>

namespace {
constexpr int kTile = 64;
constexpr int kHeads = 8;
constexpr int kQueryHeads = 32;
constexpr int kDim = 128;
constexpr int kPartialTask = 560;
constexpr int kEndEvent = 2;
using Bf = nv_bfloat16;
using GroupDivisor = flashinfer::uint_fastdiv;
__constant__ __align__(16) unsigned char group_divisor[sizeof(GroupDivisor)];

struct Part {
  int offset;
  int length;
  int causal;
  int threshold;
};
static std::vector<Part> host_parts;
static int query_count = 0;
static int query_tiles = 0;
static bool loaded = false;
static Part *device_parts = nullptr;

struct Params {
  using DTypeQ = Bf;
  using DTypeKV = Bf;
  using DTypeO = Bf;
  using IdType = int32_t;
  Bf *q, *k, *v, *o;
  float *lse;
  flashinfer::uint_fastdiv group_size;
  float sm_scale = 0.08838834764831845f;
  int window_left = -1;
  uint32_t qo_len, kv_len;
  uint32_t q_stride_n = kQueryHeads * kDim, q_stride_h = kDim;
  uint32_t k_stride_n = kHeads * kDim, k_stride_h = kDim;
  uint32_t v_stride_n = kHeads * kDim, v_stride_h = kDim;
  bool partition_kv = false;
  __host__ __device__ uint32_t get_qo_len(uint32_t) const { return qo_len; }
  __host__ __device__ uint32_t get_kv_len(uint32_t) const { return kv_len; }
};

template <flashinfer::MaskMode Mask>
using Traits = flashinfer::KernelTraits<
    Mask, kTile, 1, 4, 8, 8, 4, 1, flashinfer::PosEncodingMode::kNone,
    Bf, Bf, Bf, float, int32_t, flashinfer::DefaultAttention<false, false, false, false>>;

static_assert(Traits<flashinfer::MaskMode::kNone>::NUM_THREADS == 128);
static_assert(!Traits<flashinfer::MaskMode::kNone>::IsInvalid());
static_assert(sizeof(Traits<flashinfer::MaskMode::kNone>::SharedStorage) <=
              mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE);

template <flashinfer::MaskMode Mask>
__device__ __forceinline__ void partial(Params& p, int tile, int head) {
  extern __shared__ __align__(16) unsigned char memory[];
  auto &smem = reinterpret_cast<typename Traits<Mask>::SharedStorage&>(memory);
  dim3 tid(threadIdx.x & 31, threadIdx.x >> 5, 0);
  flashinfer::SinglePrefillWithKVCacheDevice<Traits<Mask>>(
      p, smem, tid, tile, 0, head, 1, kHeads);
}

__device__ __forceinline__ void run_part(
    Bf *q, Bf *k, Bf *v, Bf *o, float *lse, int rows,
    Part part, int part_id, int tile, int head) {
  Params p{};
  p.group_size = *reinterpret_cast<const GroupDivisor*>(group_divisor);
  p.q = q;
  p.k = k + static_cast<size_t>(part.offset) * kHeads * kDim;
  p.v = v + static_cast<size_t>(part.offset) * kHeads * kDim;
  p.o = o + static_cast<size_t>(part_id) * rows * kQueryHeads * kDim;
  p.lse = lse + static_cast<size_t>(part_id) * rows * kQueryHeads;
  p.qo_len = rows;
  p.kv_len = part.length;
  if (part.causal) {
    partial<flashinfer::MaskMode::kCausal>(p, tile, head);
  } else {
    partial<flashinfer::MaskMode::kNone>(p, tile, head);
  }
}

__global__ void dual_part(
    Bf *q, Bf *k, Bf *v, Bf *o, float *lse, int rows, Part part, int part_id) {
  run_part(q, k, v, o, lse, rows, part, part_id, blockIdx.x, blockIdx.y);
}

__global__ void publish_page(int *flag, int value) {
  asm volatile("st.release.gpu.b32 [%0], %1;" : : "l"(flag), "r"(value) : "memory");
}
} // namespace

__device__ __forceinline__ void
_execute_task(TaskDesc const *task, RuntimeConfig const &config) {
  if (task->task_type != static_cast<TaskType>(kPartialTask)) return;
  auto *parts = static_cast<Part const*>(config.hicache.parts);
  int id = task->task_metadata.request_id;
  int tile_head = task->task_metadata.merge_task_offset;
  run_part(static_cast<Bf*>(config.hicache.query), static_cast<Bf*>(config.hicache.key),
           static_cast<Bf*>(config.hicache.value), static_cast<Bf*>(config.hicache.output),
           config.hicache.lse, config.hicache.rows,
           parts[id], id, tile_head / kHeads, tile_head % kHeads);
}

static void _init_persistent_kernel(
    std::vector<FullTaskDesc> &tasks, std::vector<EventDesc> &events,
    std::vector<TaskId> &first_tasks, int num_gpus, int rank,
    std::map<std::string, void*> const&) {
  assert(num_gpus == 1 && rank == 0);
  tasks.emplace_back(TASK_TERMINATE, 0);
  tasks.emplace_back(TASK_BEGIN_TASK_GRAPH, 0);
  tasks.back().trigger_event = 1;
  for (int part = 0; part < static_cast<int>(host_parts.size()); ++part) {
    for (int tile_head = 0; tile_head < query_tiles * kHeads; ++tile_head) {
      FullTaskDesc task(static_cast<TaskType>(kPartialTask), 0);
      task.task_metadata.request_id = part;
      task.task_metadata.kv_idx = host_parts[part].threshold;
      task.task_metadata.merge_task_offset = tile_head;
      task.trigger_event = kEndEvent;
      tasks.push_back(task);
    }
  }
  events.emplace_back(EVENT_TERMINATION, 0, 0, 0);
  events.emplace_back(EVENT_LAUNCH_DEPENDENT_TASKS, 1, 2, tasks.size());
  events.emplace_back(EVENT_END_OF_TASK_GRAPH, tasks.size() - 2, 0, 0);
  first_tasks.push_back(2);
}

extern "C" int pages_init(int query, int prefix, int resident, int group) {
  if (loaded) return -1;
  if (query < 1 || resident < 0 || resident >= prefix || group < 1) return -2;
  host_parts.clear();
  if (resident > 0) host_parts.push_back({0, resident, 0, 0});
  host_parts.push_back({prefix, query, 1, 0});
  int threshold = 0;
  for (int start = resident; start < prefix; start += group) {
    host_parts.push_back({start, std::min(group, prefix - start), 0, ++threshold});
  }
  GroupDivisor div(kQueryHeads / kHeads);
  CUDA_CHECK(cudaMemcpyToSymbol(group_divisor, &div, sizeof(div)));
  query_count = query;
  query_tiles = (query * (kQueryHeads / kHeads) + kTile - 1) / kTile;
  CUDA_CHECK(cudaMalloc(&device_parts, host_parts.size() * sizeof(Part)));
  CUDA_CHECK(cudaMemcpy(device_parts, host_parts.data(), host_parts.size() * sizeof(Part),
                        cudaMemcpyHostToDevice));
  init_persistent_kernel(std::vector<void*>(11, nullptr), nullptr, 0, 80, 48, 0,
                         prefix + query, 1, -1, 0, {}, {}, {256}, {0});
  constexpr int smem = static_cast<int>(sizeof(Traits<flashinfer::MaskMode::kNone>::SharedStorage));
  CUDA_CHECK(cudaFuncSetAttribute(dual_part, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  loaded = true;
  return host_parts.size();
}

extern "C" int pages_mpk(
    void *q, void *k, void *v, void *out, void *lse, int *ready, cudaStream_t stream) {
  if (!loaded) return -1;
  global_runtime_config.hicache = {q, k, v, out, static_cast<float*>(lse),
                                  ready, device_parts, query_count};
  launch_persistent_kernel(stream);
  return cudaPeekAtLastError();
}

extern "C" int pages_wait() {
  return wait_persistent_kernel();
}

extern "C" int pages_dual(
    void *q, void *k, void *v, void *out, void *lse, int part_id, cudaStream_t stream) {
  if (!loaded || part_id < 0 || part_id >= static_cast<int>(host_parts.size())) return -1;
  constexpr int smem = static_cast<int>(sizeof(Traits<flashinfer::MaskMode::kNone>::SharedStorage));
  dual_part<<<dim3(query_tiles, kHeads), 128, smem, stream>>>(
      static_cast<Bf*>(q), static_cast<Bf*>(k), static_cast<Bf*>(v),
      static_cast<Bf*>(out), static_cast<float*>(lse), query_count, host_parts[part_id], part_id);
  return cudaPeekAtLastError();
}

extern "C" int pages_publish(int *flag, int value, cudaStream_t stream) {
  publish_page<<<1, 1, 0, stream>>>(flag, value);
  return cudaPeekAtLastError();
}

extern "C" void pages_finalize() {
  if (loaded) {
    finalize_persistent_kernel();
    CUDA_CHECK(cudaFree(device_parts));
    loaded = false;
  }
}
