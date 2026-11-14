  CUtensorMap up_map = up_weights_desc, x_map = activations_desc;
  CUtensorMap down_map = down_weights_desc, h_map = down_activations_desc;
  size_t scratch_size = scratchpad_size, shared_size = shmem_size;
  auto x = activations_in_ptr, r = router_logits_ptr;
  auto w1 = expert_weights_up_ptr, w2 = expert_weights_down_ptr;
  auto s1 = expert_scales_up_ptr, s2 = expert_scales_down_ptr;
  auto out = activations_out_ptr;
  void* temp = scratchpad_ptr;
  uint32_t count = num_tokens, k = top_k_u32;
  auto score = sf;
  void *args[] = {&x, &count, &r, &w1, &s1, &w2, &s2, &out, &temp, &scratch_size,
                  &shared_size, &k, &score, &renormalize, &up_map, &x_map, &down_map, &h_map};
  CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<const void *>(moe_kernel_topk<Dims>),
      Dims::KernelConfig::GRID_SIZE, Dims::KernelConfig::BLOCK_SIZE, args, shmem_size, stream));
