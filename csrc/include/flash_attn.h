// Shared declarations for the hand-written CUDA FlashAttention forward.
//
// Target: NVIDIA T4 (Turing, sm_75), fp16 + fp32 only (no bf16 / fp8 on Turing).
#pragma once

#include <vector>
#include <torch/extension.h>

// Launch parameters for the forward kernel. q/k/v/o are assumed contiguous and
// laid out as (B, H, N, d). `l` is the per-row logsumexp output, shape (B,H,N),
// fp32, saved so the (Triton/PyTorch) backward can recompute P without storing
// the full N x N matrix.
struct FlashFwdParams {
  const void* q;
  const void* k;
  const void* v;
  void* o;
  float* l;
  int B;
  int H;
  int N;
  int d;
  int block_m;   // query rows per block == threads per block
  int block_n;   // key/value tile width held in shared memory
  float sm_scale;
  bool causal;
};

// Defined in flash_fwd_kernel.cu, bound to Python in flash_attn.cpp.
// Returns {o, l}: o has q's dtype/shape, l is fp32 (B, H, N).
std::vector<at::Tensor> flash_fwd(at::Tensor q, at::Tensor k, at::Tensor v,
                                  bool causal, double sm_scale);
