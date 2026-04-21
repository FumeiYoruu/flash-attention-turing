"""Python interface for FlashAttention Turing extension."""

from typing import Optional, Tuple

import flash_attn_turing as flash_attn_gpu
import torch
import math


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, lse = flash_attn_gpu.fwd(
        q,
        k,
        v,
        softmax_scale,
        causal,
    )
    return out, lse


def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out, lse = [maybe_contiguous(x) for x in (dout, q, k, v, out, lse)]
    dq, dk, dv = flash_attn_gpu.bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        causal,
    )
    return dq, dk, dv


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, lse = flash_attn_gpu.varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    )
    return out, lse


def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out, lse = [maybe_contiguous(x) for x in (dout, q, k, v, out, lse)]
    dq, dk, dv = flash_attn_gpu.varlen_bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    )
    return dq, dk, dv


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        return dq, dk, dv, None, None, None


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        q, k, v = (
            qkv.select(dim=2, index=0).contiguous(),
            qkv.select(dim=2, index=1).contiguous(),
            qkv.select(dim=2, index=2).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dqkv[:, :, 0] = dq
        dqkv[:, :, 1] = dk
        dqkv[:, :, 2] = dv
        return dqkv, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        k, v = (
            kv.select(dim=2, index=0).contiguous(),
            kv.select(dim=2, index=1).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dkv[:, :, 0] = dk
        dkv[:, :, 1] = dv
        return dq, dkv, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        q, k, v = (
            qkv.select(dim=1, index=0).contiguous(),
            qkv.select(dim=1, index=1).contiguous(),
            qkv.select(dim=1, index=2).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.softmax_scale,
            ctx.causal,
        )
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dqkv[:, 0] = dq
        dqkv[:, 1] = dk
        dqkv[:, 2] = dv
        return dqkv, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        k, v = (
            kv.select(dim=1, index=0).contiguous(),
            kv.select(dim=1, index=1).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
        )
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dkv[:, 0] = dk
        dkv[:, 1] = dv
        return dq, dkv, None, None, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        k: (batch_size, seqlen_k, nheads_k, headdim)
        v: (batch_size, seqlen_k, nheads_k, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        kv: (batch_size, seqlen_k, 2, nheads_k, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total_q, nheads, headdim).
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total, nheads, headdim).
    """
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total_q, nheads, headdim).
    """
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    KV-cache inference: attend over a pre-filled KV cache plus optional new tokens.

    Arguments:
        q:             (B, seqlen_q, H, d)              float16 — new query tokens.
        k_cache:       (B, max_cache_seqlen, H_k, d)    float16 — mutable KV cache.
        v_cache:       (B, max_cache_seqlen, H_k, d)    float16 — mutable KV cache.
        cache_seqlens: (B,)                              int32   — number of valid cached
                           tokens per batch element (before this step).
        k:             (B, seqlen_knew, H_k, d)         float16 — new key tokens to append
                           into the cache (optional). If provided, they are written into
                           k_cache at positions [cache_seqlens[b] : cache_seqlens[b]+seqlen_knew].
        v:             (B, seqlen_knew, H_k, d)         float16 — new value tokens (paired with k).
        softmax_scale: float. Defaults to 1/sqrt(d).
        causal:        bool. Whether to apply causal masking (default True for inference).

    Returns:
        out: (B, seqlen_q, H, d) float16 — attention output for the new queries.

    Notes:
        - The cache tensors are updated IN-PLACE when k/v are provided.
        - Supports GQA/MQA: H must be divisible by H_k.
        - head_dim must be 64 or 128.
        - All tensors must reside on the same CUDA device and be float16.
    """
    assert q.dtype == torch.float16, "q must be float16"
    assert k_cache.dtype == torch.float16, "k_cache must be float16"
    assert v_cache.dtype == torch.float16, "v_cache must be float16"

    softmax_scale = softmax_scale or (q.shape[-1] ** -0.5)

    seqlen_knew = k.shape[1] if k is not None else 0

    # Append new K/V into the cache in-place before launching the kernel.
    # For each batch element b, write k[b] -> k_cache[b, cache_seqlens[b]:cache_seqlens[b]+seqlen_knew].
    # We use a vectorised scatter because the offsets differ per batch.
    if k is not None and v is not None and seqlen_knew > 0:
        batch_size = q.shape[0]
        # cache_seqlens is on device; bring to CPU for the Python loop (small B).
        offsets = cache_seqlens.cpu().tolist()
        for b in range(batch_size):
            start = offsets[b]
            end = start + seqlen_knew
            k_cache[b, start:end] = k[b]
            v_cache[b, start:end] = v[b]

    # Ensure contiguity for the cache tensors after the in-place update.
    q        = maybe_contiguous(q)
    k_cache  = maybe_contiguous(k_cache)
    v_cache  = maybe_contiguous(v_cache)

    out, _lse = flash_attn_gpu.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        k,          # passed so kernel knows seqlen_knew for grid sizing
        v,
        softmax_scale,
        causal,
    )
    return out
