// Hand-written CUDA FlashAttention-2 FORWARD kernel.
//
// Same algorithm as the Triton implementation (Stage 2): tile the queries,
// stream over key/value tiles held in shared memory, and maintain a running
// max (m_i), running denominator (l_i) and output accumulator (acc) with the
// online-softmax rescaling trick. The full N x N score matrix is NEVER
// materialized -> O(N) memory.
//
// Parallelization (correctness-first, deliberately simple):
//   grid  = (ceil(N / BLOCK_M), B*H)
//   block = BLOCK_M threads; thread t owns query row (blockIdx.x*BLOCK_M + t)
//           and computes that row's entire output by looping over key tiles.
//   Shared memory holds the current K tile and V tile (BLOCK_N x d) as fp32.
//
// This does NOT use warp-level tensor-core MMA (listed as a limitation in the
// README). It exists to demonstrate the algorithm in raw CUDA and to be
// validated against the same reference as every other path.
//
// Precision: inputs may be fp16 or fp32; all math accumulates in fp32.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#include <ATen/cuda/CUDAContext.h>  // getCurrentCUDAStream
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK

#include "include/flash_attn.h"

namespace {

constexpr int MAX_HEADDIM = 128;  // T4 shared-mem budget supports up to 128.

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T from_f(float x);
template <>
__device__ __forceinline__ float from_f<float>(float x) { return x; }
template <>
__device__ __forceinline__ __half from_f<__half>(float x) { return __float2half(x); }

template <typename scalar_t>
__global__ void flash_fwd_kernel(FlashFwdParams p) {
  extern __shared__ float smem[];
  float* Ks = smem;                       // [block_n * d]
  float* Vs = smem + p.block_n * p.d;     // [block_n * d]

  const int d = p.d;
  const int N = p.N;
  const int bn = p.block_n;

  const int hz = blockIdx.y;              // flattened (batch, head)
  const long bh_offset = (long)hz * N * d;
  const scalar_t* qp = reinterpret_cast<const scalar_t*>(p.q) + bh_offset;
  const scalar_t* kp = reinterpret_cast<const scalar_t*>(p.k) + bh_offset;
  const scalar_t* vp = reinterpret_cast<const scalar_t*>(p.v) + bh_offset;
  scalar_t* op = reinterpret_cast<scalar_t*>(p.o) + bh_offset;
  float* lp = p.l + (long)hz * N;

  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = row < N;

  float qreg[MAX_HEADDIM];
  float acc[MAX_HEADDIM];
  if (valid) {
    for (int x = 0; x < d; ++x) qreg[x] = to_f(qp[(long)row * d + x]);
  }
  for (int x = 0; x < d; ++x) acc[x] = 0.f;

  float m_i = -INFINITY;
  float l_i = 0.f;

  // For causal masking, the largest key index any row in this block can attend
  // to is (block base + BLOCK_M - 1); cap the key sweep there.
  const int n_end =
      p.causal ? min(N, blockIdx.x * (int)blockDim.x + (int)blockDim.x) : N;

  for (int start_n = 0; start_n < n_end; start_n += bn) {
    // Cooperatively stage K and V tiles into shared memory (as fp32).
    for (int idx = threadIdx.x; idx < bn * d; idx += blockDim.x) {
      const int jj = idx / d;     // local key index within tile
      const int dd = idx % d;
      const int kg = start_n + jj;  // global key index
      float kv_k = 0.f, kv_v = 0.f;
      if (kg < N) {
        kv_k = to_f(kp[(long)kg * d + dd]);
        kv_v = to_f(vp[(long)kg * d + dd]);
      }
      Ks[idx] = kv_k;
      Vs[idx] = kv_v;
    }
    __syncthreads();

    if (valid) {
      const int jmax = min(bn, N - start_n);
      for (int j = 0; j < jmax; ++j) {
        const int kg = start_n + j;
        if (p.causal && kg > row) break;  // keys ascend; rest are masked too

        float s = 0.f;
        for (int x = 0; x < d; ++x) s += qreg[x] * Ks[j * d + x];
        s *= p.sm_scale;

        const float m_new = fmaxf(m_i, s);
        const float p_ij = __expf(s - m_new);
        const float corr = __expf(m_i - m_new);
        l_i = l_i * corr + p_ij;
        for (int x = 0; x < d; ++x) acc[x] = acc[x] * corr + p_ij * Vs[j * d + x];
        m_i = m_new;
      }
    }
    __syncthreads();  // protect smem before the next tile overwrites it
  }

  if (valid) {
    const float inv = 1.f / l_i;
    for (int x = 0; x < d; ++x) op[(long)row * d + x] = from_f<scalar_t>(acc[x] * inv);
    lp[row] = m_i + logf(l_i);  // logsumexp for the backward pass
  }
}

template <typename scalar_t>
void launch(FlashFwdParams& p, cudaStream_t stream) {
  dim3 grid((p.N + p.block_m - 1) / p.block_m, p.B * p.H);
  dim3 block(p.block_m);
  size_t smem = (size_t)2 * p.block_n * p.d * sizeof(float);
  flash_fwd_kernel<scalar_t><<<grid, block, smem, stream>>>(p);
}

}  // namespace

std::vector<at::Tensor> flash_fwd(at::Tensor q, at::Tensor k, at::Tensor v,
                                  bool causal, double sm_scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(q.dim() == 4, "expected (B, H, N, d)");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, v must share dtype");
  TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf,
              "only fp16/fp32 supported (T4 has no bf16/fp8)");

  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();

  const int B = q.size(0), H = q.size(1), N = q.size(2), d = q.size(3);
  TORCH_CHECK(d <= MAX_HEADDIM, "head_dim must be <= 128 on T4");

  auto o = at::empty_like(q);
  auto l = at::empty({B, H, N}, q.options().dtype(at::kFloat));

  FlashFwdParams p;
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.o = o.data_ptr();
  p.l = l.data_ptr<float>();
  p.B = B; p.H = H; p.N = N; p.d = d;
  p.block_m = 64;
  p.block_n = (d <= 64) ? 64 : 32;  // keep 2*block_n*d*4 bytes <= ~48KB smem
  p.sm_scale = (float)sm_scale;
  p.causal = causal;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == at::kFloat) {
    launch<float>(p, stream);
  } else {
    launch<__half>(p, stream);
  }
  C10_CUDA_CHECK(cudaGetLastError());
  return {o, l};
}
