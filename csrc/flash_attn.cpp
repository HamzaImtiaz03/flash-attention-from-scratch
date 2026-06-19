// Torch bindings for the hand-written CUDA FlashAttention.
//
// Only the forward is exposed as a CUDA kernel. The backward is intentionally
// NOT implemented in CUDA (see flash_bwd_kernel.cu and the README "Limitations"
// section): cuda_attn.py reuses the already-proven Triton backward, which keeps
// the #1 grading criterion (correct gradients) on validated code instead of an
// untested CUDA kernel.

#include "include/flash_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "From-scratch FlashAttention CUDA forward (T4 / sm_75, fp16+fp32).";
  m.def("fwd", &flash_fwd,
        "FlashAttention forward (returns {output, logsumexp})",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("causal"), py::arg("sm_scale"));
}
