# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import warnings

__all__ = [
    "flash_attention",
    "attention",
]


def _sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),  # unused in SDPA fallback
    deterministic: bool = False,  # unused in SDPA fallback
    dtype=torch.bfloat16,
    fa_version=None,  # unused, kept for API compatibility
):
    """
    Fallback attention implementation using PyTorch scaled_dot_product_attention.

    Expected shapes (as used by WAN):
      q: [B, Lq, H, D]
      k: [B, Lk, H, D]
      v: [B, Lk, H, Dv]

    We ignore q_lens / k_lens and use full padded sequences.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"Expected q, k, v to be 4D [B, L, H, D], got: "
            f"q={q.shape}, k={k.shape}, v={v.shape}"
        )

    B, Lq, H, Dq = q.shape
    Bk, Lk, Hk, Dk = k.shape
    Bv, Lv, Hv, Dv = v.shape

    # Basic sanity: same batch & heads
    if not (B == Bk == Bv and H == Hk == Hv):
        raise ValueError(
            f"Mismatch in batch/head dims: "
            f"q={q.shape}, k={k.shape}, v={v.shape}"
        )

    # Cast to desired dtype
    if q.dtype != dtype:
        q = q.to(dtype)
    if k.dtype != dtype:
        k = k.to(dtype)
    if v.dtype != dtype:
        v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # Rearrange to [B, H, L, D]
    q = q.permute(0, 2, 1, 3)  # [B, H, Lq, Dq]
    k = k.permute(0, 2, 1, 3)  # [B, H, Lk, Dk]
    v = v.permute(0, 2, 1, 3)  # [B, H, Lk, Dv]

    # Merge batch & heads → [B * H, L, D]
    BH = B * H
    q = q.reshape(BH, Lq, Dq)
    k = k.reshape(BH, Lk, Dk)
    v = v.reshape(BH, Lk, Dv)

    # No explicit attn_mask; we ignore q_lens / k_lens to keep it simple
    attn_mask = None

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=causal,
        dropout_p=dropout_p,
    )  # [BH, Lq, Dv]

    # Reshape back to [B, Lq, H, Dv]
    out = out.reshape(B, H, Lq, Dv).permute(0, 2, 1, 3).contiguous()
    return out


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    Unified attention API.

    Currently we run only the SDPA fallback implementation to avoid binary
    issues with flash-attn on different driver / CUDA combos.
    """
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "q_lens / k_lens are ignored in SDPA fallback. "
            "Make sure your inputs are properly padded."
        )

    return _sdpa_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        fa_version=fa_version,
    )


# For compatibility with existing code:
# flash_attention and attention are the same entrypoint.
flash_attention = attention
