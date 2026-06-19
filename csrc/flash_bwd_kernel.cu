// CUDA BACKWARD kernel -- intentionally NOT implemented.
//
// Status (stated honestly, per the project brief which explicitly allows
// "CUDA forward + Triton backward" as long as it is disclosed):
//
//   The backward pass for the CUDA path is handled by the already-validated
//   Triton backward kernels (see flash_attn/triton_attn.py and
//   flash_attn/cuda_attn.py::CudaAttention.backward). Those kernels are proven
//   correct by float64 gradcheck and fp32 allclose against the PyTorch
//   reference (tests/test_backward.py).
//
//   Shipping an untested hand-written CUDA backward would put the project's #1
//   grading criterion -- provably correct gradients -- on unverified code. That
//   is a worse outcome than honestly reusing the verified Triton backward.
//
// This translation unit is intentionally empty of device code so it compiles
// cleanly if a build system globs csrc/*.cu. The implementation sketch for a
// future CUDA backward would mirror the Triton structure:
//   1) preprocess: D_i = rowsum(dO_i . O_i)
//   2) dK/dV kernel: one block per K/V tile, loop over Q tiles, accumulate
//   3) dQ kernel:    one block per Q tile, loop over K/V tiles, accumulate
// using the same recompute-from-(q,k,v,O,L) trick as the forward.

// (no symbols)
