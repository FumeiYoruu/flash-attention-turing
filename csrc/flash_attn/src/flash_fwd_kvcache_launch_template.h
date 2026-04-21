#pragma once

#include "flash.h"
#include "flash_fwd_kvcache_kernel.h"
#include "static_switch.h"
#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

// ---------------------------------------------------------------------------
// CUDA global kernel — thin wrapper that calls compute_attn_kvcache
// ---------------------------------------------------------------------------
template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
__global__ __launch_bounds__(256)
void flash_fwd_kvcache_kernel(
    const half_t* __restrict__ q,
    const half_t* __restrict__ kcache,
    const half_t* __restrict__ vcache,
    const int64_t kcache_batch_stride,
    const int64_t kcache_row_stride,
    const int64_t kcache_head_stride,
    const int64_t vcache_batch_stride,
    const int64_t vcache_row_stride,
    const int64_t vcache_head_stride,
    const half_t* __restrict__ knew,
    const half_t* __restrict__ vnew,
    const int64_t knew_batch_stride,
    const int64_t knew_row_stride,
    const int64_t knew_head_stride,
    const int64_t vnew_batch_stride,
    const int64_t vnew_row_stride,
    const int64_t vnew_head_stride,
    half_t* __restrict__ o,
    float*  __restrict__ l,
    const int* __restrict__ cache_seqlens,
    const int seqlen_knew,
    const int batch_size,
    const int max_seqlen_q,
    const int num_heads,
    const int num_heads_k,
    const int h_h_k_ratio,
    const int head_dim,
    const float softmax_scale)
{
    compute_attn_kvcache<Kernel_traits, Is_causal, Is_even_MN>(
        q,
        kcache, vcache,
        kcache_batch_stride, kcache_row_stride, kcache_head_stride,
        vcache_batch_stride, vcache_row_stride, vcache_head_stride,
        knew, vnew,
        knew_batch_stride, knew_row_stride, knew_head_stride,
        vnew_batch_stride, vnew_row_stride, vnew_head_stride,
        o, l,
        cache_seqlens, seqlen_knew,
        batch_size, max_seqlen_q,
        num_heads, num_heads_k, h_h_k_ratio, head_dim,
        softmax_scale);
}

// ---------------------------------------------------------------------------
// run_flash_fwd_kvcache — host-side launch logic
// ---------------------------------------------------------------------------
template <typename Kernel_traits, bool Is_causal>
void run_flash_fwd_kvcache(Flash_fwd_kvcache_params &params) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;

    // is_even_MN: the K dimension is ragged per-batch (cache_seqlens differ),
    // so we conservatively always use the masked path.
    // Q is always dense (no varlen on query for kvcache inference).
    const bool is_even_MN = (params.seqlen_q % kBlockM == 0) &&
                             (params.seqlen_k % kBlockN == 0) &&
                             (params.seqlen_knew == 0);

    dim3 dimGrid(num_m_block, params.b, params.h);
    dim3 dimBlock(256);
    int  maxbytes = 65536;

    BOOL_SWITCH(is_even_MN, Is_even_MN, [&] {
        cudaFuncSetAttribute(
            flash_fwd_kvcache_kernel<Kernel_traits, Is_causal, Is_even_MN>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        flash_fwd_kvcache_kernel<Kernel_traits, Is_causal, Is_even_MN>
            <<<dimGrid, dimBlock, maxbytes>>>(
                params.q_ptr,
                params.kcache_ptr, params.vcache_ptr,
                params.kcache_batch_stride, params.kcache_row_stride, params.kcache_head_stride,
                params.vcache_batch_stride, params.vcache_row_stride, params.vcache_head_stride,
                params.knew_ptr,  params.vnew_ptr,
                params.knew_batch_stride, params.knew_row_stride, params.knew_head_stride,
                params.vnew_batch_stride, params.vnew_row_stride, params.vnew_head_stride,
                params.o_ptr, params.l_ptr,
                params.cache_seqlens,
                params.seqlen_knew,
                params.b,
                params.seqlen_q,
                params.h,
                params.h_k,
                params.h_h_k_ratio,
                params.d,
                params.softmax_scale);
    });
}

// ---------------------------------------------------------------------------
// Concrete instantiations for hdim64 and hdim128
// ---------------------------------------------------------------------------
template <bool Is_causal>
void run_mha_fwd_kvcache_hdim64(Flash_fwd_kvcache_params &params) {
    constexpr int Headdim = 64;
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kNWarps = 8;
    run_flash_fwd_kvcache<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template <bool Is_causal>
void run_mha_fwd_kvcache_hdim128(Flash_fwd_kvcache_params &params) {
    constexpr int Headdim = 128;
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 64;
    constexpr int kNWarps = 8;
    run_flash_fwd_kvcache<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}
