#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void check_driver(CUresult status) {
  if (status == CUDA_SUCCESS) return;
  const char* text = nullptr;
  cuGetErrorString(status, &text);
  throw std::runtime_error(text ? text : "CUDA driver failed");
}

struct Plan {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;
  std::vector<cudaEvent_t> starts, stops;

  ~Plan() {
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
    for (auto e : starts) if (e) cudaEventDestroy(e);
    for (auto e : stops) if (e) cudaEventDestroy(e);
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
  }
};

cudaGraphNode_t transplant(cudaGraph_t dst, cudaGraphNode_t original,
                           const std::vector<cudaGraphNode_t>& predecessors) {
  auto inputs = predecessors.data();
  auto n = predecessors.size();
  cudaGraphNode_t result;
  cudaGraphNodeType type;
  check(cudaGraphNodeGetType(original, &type));
  switch (type) {
    case cudaGraphNodeTypeKernel: {
      CUDA_KERNEL_NODE_PARAMS p{};
      check_driver(cuGraphKernelNodeGetParams(original, &p));
      check_driver(cuGraphAddKernelNode(&result, dst, inputs, n, &p));
      break;
    }
    case cudaGraphNodeTypeMemcpy: {
      cudaMemcpy3DParms p{};
      check(cudaGraphMemcpyNodeGetParams(original, &p));
      check(cudaGraphAddMemcpyNode(&result, dst, inputs, n, &p));
      break;
    }
    case cudaGraphNodeTypeMemset: {
      cudaMemsetParams p{};
      check(cudaGraphMemsetNodeGetParams(original, &p));
      check(cudaGraphAddMemsetNode(&result, dst, inputs, n, &p));
      break;
    }
    case cudaGraphNodeTypeEmpty:
      check(cudaGraphAddEmptyNode(&result, dst, inputs, n));
      break;
    default:
      throw std::runtime_error("Unsupported captured node type " + std::to_string(type));
  }
  return result;
}

cudaGraphNode_t append(cudaGraph_t dst, cudaGraph_t src,
                       const std::vector<cudaGraphNode_t>& dependencies) {
  size_t length = 0;
  check(cudaGraphGetNodes(src, nullptr, &length));
  std::vector<cudaGraphNode_t> originals(length);
  check(cudaGraphGetNodes(src, originals.data(), &length));
  std::unordered_map<cudaGraphNode_t, cudaGraphNode_t> added;
  while (added.size() < length) {
    auto previous = added.size();
    for (auto node : originals) {
      if (added.count(node)) continue;
      size_t num_parents = 0;
      check(cudaGraphNodeGetDependencies(node, nullptr, nullptr, &num_parents));
      std::vector<cudaGraphNode_t> parents(num_parents);
      check(cudaGraphNodeGetDependencies(node, parents.data(), nullptr, &num_parents));
      std::vector<cudaGraphNode_t> inputs;
      for (auto p : parents) {
        auto child = added.find(p);
        if (child != added.end()) inputs.push_back(child->second);
      }
      if (inputs.size() != num_parents) continue;
      if (inputs.empty()) inputs = dependencies;
      added[node] = transplant(dst, node, inputs);
    }
    if (added.size() == previous) throw std::runtime_error("Captured graph has a cycle");
  }
  std::vector<cudaGraphNode_t> leaves;
  for (auto node : originals) {
    size_t outgoing = 0;
    check(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &outgoing));
    if (!outgoing) leaves.push_back(added.at(node));
  }
  if (leaves.size() == 1) return leaves[0];
  cudaGraphNode_t join;
  const auto& inputs = leaves.empty() ? dependencies : leaves;
  check(cudaGraphAddEmptyNode(&join, dst, inputs.data(), inputs.size()));
  return join;
}
}  // namespace

extern "C" void* plan_create(int count, const uintptr_t* bodies, const int* order,
                             const int* offsets, const int* prereqs, int traced) {
  try {
    auto p = std::make_unique<Plan>();
    check(cudaGraphCreate(&p->graph, 0));
    check(cudaEventCreate(&p->begin));
    check(cudaEventCreate(&p->end));
    p->starts.resize(count);
    p->stops.resize(count);
    std::vector<cudaGraphNode_t> ends(count);
    for (int k = 0; k < count; ++k) {
      int i = order[k];
      std::vector<cudaGraphNode_t> parents;
      for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
        if (!ends[prereqs[j]]) throw std::runtime_error("Non-topological creation order");
        parents.push_back(ends[prereqs[j]]);
      }
      if (traced) {
        cudaGraphNode_t node;
        check(cudaEventCreate(&p->starts[i]));
        check(cudaGraphAddEventRecordNode(&node, p->graph, parents.data(), parents.size(), p->starts[i]));
        parents = {node};
      }
      auto end = append(p->graph, reinterpret_cast<cudaGraph_t>(bodies[i]), parents);
      if (traced) {
        check(cudaEventCreate(&p->stops[i]));
        check(cudaGraphAddEventRecordNode(&ends[i], p->graph, &end, 1, p->stops[i]));
      } else {
        ends[i] = end;
      }
    }
    check(cudaGraphInstantiateWithFlags(&p->exec, p->graph, 0));
    return p.release();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "plan_create: %s\n", e.what());
    return nullptr;
  }
}

extern "C" int plan_run(void* handle, uintptr_t stream, float* elapsed, float* trace) {
  auto& p = *static_cast<Plan*>(handle);
  auto s = reinterpret_cast<cudaStream_t>(stream);
  try {
    check(cudaEventRecord(p.begin, s));
    check(cudaGraphLaunch(p.exec, s));
    check(cudaEventRecord(p.end, s));
    check(cudaEventSynchronize(p.end));
    check(cudaEventElapsedTime(elapsed, p.begin, p.end));
    if (trace) {
      for (size_t i = 0; i < p.starts.size(); ++i) {
        check(cudaEventElapsedTime(&trace[2 * i], p.begin, p.starts[i]));
        check(cudaEventElapsedTime(&trace[2 * i + 1], p.begin, p.stops[i]));
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "plan_run: %s\n", e.what());
    return 1;
  }
}

extern "C" void plan_destroy(void* handle) {
  delete static_cast<Plan*>(handle);
}

// Peak possible resident-block pressure over the kernels in a logical stage.
// This is a launch-resource proxy. It uses occupancy limits and grid sizes,
// and does not sample runtime activity.
extern "C" float fragment_pressure(uintptr_t raw, int sm_count) {
  try {
    auto graph = reinterpret_cast<cudaGraph_t>(raw);
    size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph, nodes.data(), &count));
    double peak = 0.;
    for (auto node : nodes) {
      cudaGraphNodeType type;
      check(cudaGraphNodeGetType(node, &type));
      if (type != cudaGraphNodeTypeKernel) continue;
      CUDA_KERNEL_NODE_PARAMS params{};
      check_driver(cuGraphKernelNodeGetParams(node, &params));
      int block_size = params.blockDimX * params.blockDimY * params.blockDimZ;
      int resident_blocks = 0;
      check_driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &resident_blocks, params.func, block_size, params.sharedMemBytes));
      auto resident_capacity = double(resident_blocks) * sm_count;
      auto grid = double(params.gridDimX) * params.gridDimY * params.gridDimZ;
      if (resident_capacity > 0) peak = std::max(peak, std::min(1., grid / resident_capacity));
    }
    return static_cast<float>(peak);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fragment_pressure: %s\n", e.what());
    return -1.;
  }
}
