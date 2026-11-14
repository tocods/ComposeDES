  if constexpr (Phase == 1) {
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) spec->cut_used = shmem->expert_count;
      for (unsigned e = threadIdx.x; e < shmem->expert_count; e += blockDim.x) {
        unsigned id = shmem->experts[e].id;
        spec->cut_experts[e] = shmem->experts[e];
        spec->cut_slot[id] = u_tma->expert_slot_start[id];
        spec->cut_count[id] = u_tma->expert_routed_count[id];
        for (unsigned b = 0; b < batch_size; ++b) spec->cut_rank[id][b] = u_tma->down_rank[id][b];
      }
    }
  }
  }
@@restore
  if constexpr (Phase == 2) {
    if (threadIdx.x == 0) shmem->expert_count = spec->cut_used;
    for (unsigned e = threadIdx.x; e < spec->cut_used; e += blockDim.x) {
      unsigned id = spec->cut_experts[e].id;
      shmem->experts[e] = spec->cut_experts[e];
      u_tma->expert_slot_start[id] = spec->cut_slot[id];
      u_tma->expert_routed_count[id] = spec->cut_count[id];
      for (unsigned b = 0; b < batch_size; ++b) u_tma->down_rank[id][b] = spec->cut_rank[id][b];
    }
    __syncthreads();
  }
