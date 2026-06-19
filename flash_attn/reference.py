"""Ground-truth scaled-dot-product attention (Stage 1).

This is the correctness oracle for the whole project. It is deliberately the
naive O(N^2)-memory implementation: it materializes the full N x N score matrix
so that there is zero ambiguity about what "the right answer" is. Every other
implementation (Triton, CUDA) is checked against this one.

Layout convention used everywhere in the repo: q, k, v are (B, H, N, d).
"""

import torch


def attention(q, k, v, causal: bool, sm_scale: float):
    """Standard scaled-dot-product attention, full N x N scores.

    Args:
        q, k, v: tensors of shape (B, H, N, d).
        causal:  if True, apply an upper-triangular -inf mask so position i can
                 only attend to positions j <= i.
        sm_scale: scaling applied to the QK^T scores (typically 1/sqrt(d)).

    Returns:
        Output tensor of shape (B, H, N, d), same dtype as q.

    Notes:
        - softmax is computed in float32 for numerical stability even when the
          inputs are fp16, then cast back. This mirrors what the flash kernels
          do internally (fp32 accumulation) and keeps the oracle trustworthy as
          a comparison target for fp16 runs.
        - This intentionally costs O(N^2) memory; it WILL OOM at large N. That is
          the whole point of the memory benchmark in Stage 5.
    """
    # (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
    scores = torch.matmul(q, k.transpose(-2, -1)).float() * sm_scale

    if causal:
        N_q, N_k = q.shape[-2], k.shape[-2]
        # mask[i, j] = True where j > i (strictly upper triangular) -> disallowed.
        # diagonal offset (N_k - N_q) aligns the mask to the bottom-right so it is
        # correct even if N_q != N_k (here they are equal, but this is the safe form).
        mask = torch.triu(
            torch.ones(N_q, N_k, dtype=torch.bool, device=q.device),
            diagonal=1 + (N_k - N_q),
        )
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn.to(v.dtype), v)
    return out
