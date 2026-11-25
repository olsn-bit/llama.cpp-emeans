#pragma once

#include "common.cuh"

bool ggml_cuda_emeans_has(const ggml_tensor * src0);
bool ggml_cuda_emeans_try_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_emeans_try_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_emeans_try_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
