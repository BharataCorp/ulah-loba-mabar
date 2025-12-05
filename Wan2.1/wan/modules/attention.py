# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # =============================================================
    # FlashAttention v3
    # =============================================================
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        try:
            x = flash_attn_interface.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic
            )[0].unflatten(0, (b, lq))
            return x.type(out_dtype)
        except Exception as e:
            warnings.warn(f"[WAN] FlashAttn v3 failed, fallback to v2 or standard attention. Err={e}")

    # =============================================================
    # FlashAttention v2
    # =============================================================
    if FLASH_ATTN_2_AVAILABLE:
        try:
            x = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic
            ).unflatten(0, (b, lq))
            return x.type(out_dtype)
        except Exception as e:
            warnings.warn(f"[WAN] FlashAttn v2 failed, fallback to scaled_dot_product_attention. Err={e}")

    # =============================================================
    # Fallback mode (Torch Built-in)
    # =============================================================
    # reshape to [B, heads, seq, dim]
    q2 = q.view(b, lq, -1).transpose(1, 2)
    k2 = k.view(b, lk, -1).transpose(1, 2)
    v2 = v.view(b, lk, -1).transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(
        q2, k2, v2,
        attn_mask=None,
        is_causal=causal,
        dropout_p=dropout_p
    )

    out = out.transpose(1, 2).contiguous()
    return out.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # Try FlashAttention first if available
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        try:
            return flash_attention(
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
                version=fa_version,
            )
        except Exception as e:
            warnings.warn(f"[WAN] flash_attention call failed → fallback. err={e}")

    # =============================================================
    # Fallback when flash-attn not working
    # =============================================================
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask ignored in fallback attention. Possible minor slowdown.',
        )

    q2 = q.transpose(1, 2).to(dtype)
    k2 = k.transpose(1, 2).to(dtype)
    v2 = v.transpose(1, 2).to(dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q2, k2, v2,
        attn_mask=None,
        is_causal=causal,
        dropout_p=dropout_p
    )

    out = out.transpose(1, 2).contiguous()
    return out
