#pragma once

#include "common.cuh"

bool ggml_cuda_is_resident_moe(const ggml_tensor *op);
void ggml_cuda_resident_moe(ggml_backend_cuda_context &ctx, ggml_tensor *dst);
