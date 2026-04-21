#pragma once

#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    half_t *__restrict__ q_ptr;
    half_t *__restrict__ k_ptr;
    half_t *__restrict__ v_ptr;

    // index_t q_batch_stride;
    // index_t k_batch_stride;
    // index_t v_batch_stride;
    // index_t q_row_stride;
    // index_t k_row_stride;
    // index_t v_row_stride;
    // index_t q_head_stride;
    // index_t k_head_stride;
    // index_t v_head_stride;

    // The number of heads.
    int h;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_k;
    int h_h_k_ratio;

    // Optional varlen cumulative sequence lengths (device pointers, int32).
    int *__restrict__ cu_seqlens_q;
    int *__restrict__ cu_seqlens_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    half_t * __restrict__ o_ptr;


    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;


    // The pointer to the softmax sum.
    float * __restrict__ l_ptr;
    float softmax_scale;
    int b, seqlen_q, seqlen_k, d;
    bool is_causal;
};


struct Flash_bwd_params : public Flash_fwd_params {

    half_t *__restrict__ do_ptr;
    half_t *__restrict__ dq_ptr;
    half_t *__restrict__ dk_ptr;
    half_t *__restrict__ dv_ptr;
    float *__restrict__ do_o_ptr;

    // index_t do_batch_stride;
    // index_t do_row_stride;
    // index_t do_head_stride;
    // index_t dq_batch_stride;
    // index_t dk_batch_stride;
    // index_t dv_batch_stride;
    // index_t dq_row_stride;
    // index_t dk_row_stride;
    // index_t dv_row_stride;
    // index_t dq_head_stride;
    // index_t dk_head_stride;
    // index_t dv_head_stride;

};


// Parameters for KV-cache inference.
// Q shape:       (B, seqlen_q,     H,   d)  — new query tokens
// K/V cache:     (B, max_cache_seqlen, H_k, d)  — pre-allocated cache
// new K/V:       (B, seqlen_knew,  H_k, d)  — tokens to append (optional)
// cache_seqlens: (B,) int32 — how many valid tokens are already in the cache per batch
//
// Before the kernel runs the C++ wrapper appends knew/vnew into the cache in-place
// at positions [cache_seqlens[b] .. cache_seqlens[b]+seqlen_knew).
// The kernel then attends over total seqlen_k = cache_seqlens[b] + seqlen_knew.
struct Flash_fwd_kvcache_params : public Flash_fwd_params {
    // KV cache tensors — shape (B, max_cache_seqlen, H_k, d), contiguous
    half_t *__restrict__ kcache_ptr;
    half_t *__restrict__ vcache_ptr;
    index_t kcache_batch_stride;   // stride over batch dim (elements)
    index_t kcache_row_stride;     // stride over seq dim  (= H_k * d)
    index_t kcache_head_stride;    // stride over head dim (= d)
    index_t vcache_batch_stride;
    index_t vcache_row_stride;
    index_t vcache_head_stride;

    // Per-batch actual used cache lengths, device pointer, int32, shape (B,)
    int *__restrict__ cache_seqlens;

    // New K/V tokens to append — shape (B, seqlen_knew, H_k, d), may be nullptr
    half_t *__restrict__ knew_ptr;
    half_t *__restrict__ vnew_ptr;
    index_t knew_batch_stride;
    index_t knew_row_stride;
    index_t knew_head_stride;
    index_t vnew_batch_stride;
    index_t vnew_row_stride;
    index_t vnew_head_stride;
    int seqlen_knew;           // 0 if no new tokens to append
};

template<int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params);

template<int Headdim> void run_mha_fwd_kvcache_(Flash_fwd_kvcache_params &params);

template<int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params);
