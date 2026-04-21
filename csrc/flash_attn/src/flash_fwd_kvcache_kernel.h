// KV-cache inference forward kernel for Turing (SM75).
//
// Each query token attends over:
//   K = [ kcache[b, 0..cache_seqlens[b]-1, h_k, :],
//          knew[b,  0..seqlen_knew-1,        h_k, :] ]
//   (concatenated logically; physically loaded from two separate tensors)
//
// Is_causal=true applies a causal mask so query i can only attend to keys <= i
// relative to the full sequence position (cache_seqlens[b] + seqlen_q_new - 1
// for the last query).  The causal offset equals total_seqlen_k - seqlen_q.
//
// The kernel signature mirrors flash_fwd_kernel.h / compute_attn_1rowblock but
// reads K/V from one of two sources depending on the n_block index.

#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <float.h>
#include <torch/extension.h>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "mask.h"

using namespace cute;

// ---------------------------------------------------------------------------
// compute_attn_kvcache_1rowblock
// ---------------------------------------------------------------------------
// Template parameters:
//   Kernel_traits  — tile/warp/atom config from kernel_traits.h
//   Is_causal      — whether to apply causal mask
//   Is_even_MN     — whether seqlen_q % kBlockM == 0 && total_seqlen_k % kBlockN == 0
//
// Per-thread arguments mirror those of compute_attn_1rowblock in the
// standard forward kernel, with extra KV-cache / new-KV pointers.
template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_attn_kvcache_1rowblock(
    // Query: (total_q, H, d) packed or (B, seqlen_q, H, d) standard
    const half_t* __restrict__ q,
    // KV cache: (B, max_cache_seqlen, H_k, d)
    const half_t* __restrict__ kcache,
    const half_t* __restrict__ vcache,
    const int64_t kcache_batch_stride,  // in elements
    const int64_t kcache_row_stride,
    const int64_t kcache_head_stride,
    const int64_t vcache_batch_stride,
    const int64_t vcache_row_stride,
    const int64_t vcache_head_stride,
    // New K/V tokens (may be nullptr when seqlen_knew == 0)
    const half_t* __restrict__ knew,
    const half_t* __restrict__ vnew,
    const int64_t knew_batch_stride,
    const int64_t knew_row_stride,
    const int64_t knew_head_stride,
    const int64_t vnew_batch_stride,
    const int64_t vnew_row_stride,
    const int64_t vnew_head_stride,
    // Output: same shape as Q
    half_t* __restrict__ o,
    float*  __restrict__ l,
    // Sequence length metadata
    const int *__restrict__ cache_seqlens,  // (B,) actual used cache length
    const int seqlen_knew,                  // new tokens appended this step
    const int batch_size,
    const int max_seqlen_q,
    const int num_heads,
    const int num_heads_k,
    const int h_h_k_ratio,
    const int head_dim,
    const float softmax_scale,
    // Block indices (set by compute_attn_kvcache dispatcher)
    const int bidb,
    const int bidh,
    const int m_block)
{
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // Per-batch lengths
    const int cache_seqlen_b = cache_seqlens[bidb];
    const int total_seqlen_k = cache_seqlen_b + seqlen_knew;  // full K context
    const int seqlen_q       = max_seqlen_q;                  // always dense (no varlen on Q for kvcache)

    if (m_block * kBlockM >= seqlen_q) { return; }

    // --- Q tensor (dense, no varlen on query side) ---
    // Shape (B, seqlen_q, H, d)
    const int q_batch_offset = bidb * seqlen_q * num_heads * head_dim;
    Tensor mQ = make_tensor(make_gmem_ptr(q + q_batch_offset),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    // --- Output tensor ---
    Tensor mO = make_tensor(make_gmem_ptr(o + q_batch_offset),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    // --- LSE tensor ---
    Tensor mL = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(l)),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads, max_seqlen_q, Int<1>{}));
    Tensor gL = local_tile(mL(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(m_block));

    // --- KV cache tensors (indexed by n_block) ---
    // Cache K:  base pointer for this (batch, head_k)
    const int h_k = bidh / h_h_k_ratio;
    const half_t* kcache_bh = kcache
        + (int64_t)bidb * kcache_batch_stride
        + (int64_t)h_k  * kcache_head_stride;
    const half_t* vcache_bh = vcache
        + (int64_t)bidb * vcache_batch_stride
        + (int64_t)h_k  * vcache_head_stride;

    // We build a fake "full" tensor over total_seqlen_k with the cache row stride.
    // n_block < ceil(cache_seqlen_b / kBlockN)  → loads from cache
    // n_block >= that threshold                  → loads from knew/vnew
    const int n_cache_blocks = (cache_seqlen_b + kBlockN - 1) / kBlockN;

    // Tensor views for cache K/V (rows 0..cache_seqlen_b-1)
    Tensor mKcache = make_tensor(make_gmem_ptr(kcache_bh),
                                 make_shape(cache_seqlen_b, head_dim),
                                 make_stride(kcache_row_stride, Int<1>{}));
    Tensor gKcache = local_tile(mKcache, Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_coord(_, 0));

    Tensor mVcache = make_tensor(make_gmem_ptr(vcache_bh),
                                 make_shape(cache_seqlen_b, head_dim),
                                 make_stride(vcache_row_stride, Int<1>{}));
    Tensor gVcache = local_tile(mVcache, Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_coord(_, 0));

    // Tensor views for new K/V (rows 0..seqlen_knew-1)
    const half_t* knew_bh  = (knew != nullptr)
        ? knew  + (int64_t)bidb * knew_batch_stride  + (int64_t)h_k * knew_head_stride
        : nullptr;
    const half_t* vnew_bh  = (vnew != nullptr)
        ? vnew  + (int64_t)bidb * vnew_batch_stride  + (int64_t)h_k * vnew_head_stride
        : nullptr;

    // new-token K tensor (only valid if seqlen_knew > 0)
    // Shape (seqlen_knew, head_dim)
    Tensor mKnew = (seqlen_knew > 0 && knew_bh != nullptr)
        ? make_tensor(make_gmem_ptr(knew_bh),
                      make_shape(seqlen_knew, head_dim),
                      make_stride(knew_row_stride, Int<1>{}))
        : make_tensor(make_gmem_ptr(static_cast<const half_t*>(nullptr)),
                      make_shape(0, head_dim),
                      make_stride(knew_row_stride, Int<1>{}));
    Tensor gKnew = local_tile(mKnew, Shape<Int<kBlockN>, Int<kHeadDim>>{},
                              make_coord(_, 0));

    Tensor mVnew = (seqlen_knew > 0 && vnew_bh != nullptr)
        ? make_tensor(make_gmem_ptr(vnew_bh),
                      make_shape(seqlen_knew, head_dim),
                      make_stride(vnew_row_stride, Int<1>{}))
        : make_tensor(make_gmem_ptr(static_cast<const half_t*>(nullptr)),
                      make_shape(0, head_dim),
                      make_stride(vnew_row_stride, Int<1>{}));
    Tensor gVnew = local_tile(mVnew, Shape<Int<kBlockN>, Int<kHeadDim>>{},
                              make_coord(_, 0));

    // --- Shared memory ---
    extern __shared__ char smem_[];
    Tensor sQ  = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])),
                             typename Kernel_traits::SmemLayoutQ{});
    Tensor sK  = make_tensor(sQ.data() + kBlockM * kHeadDim, typename Kernel_traits::SmemLayoutK{});
    Tensor sV  = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVTransposed{});
    Tensor sO  = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])),
                             typename Kernel_traits::SmemLayoutQ{});

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int thread_row      = warp_id * 16 + lane_id / 4;
    const int global_row_offset = m_block * kBlockM;

    float rM_old[2] = {-FLT_MAX, -FLT_MAX};
    float rM[2]     = {0.0f};
    float rL_old[2] = {0.0f};
    float rL[2]     = {0.0f};
    float rD[2]     = {0.0f};

    // Lane-group masks for warp-level reductions (same as standard fwd kernel)
    unsigned mask;
    if      (lane_id <  4) mask = 0x0000000F;
    else if (lane_id <  8) mask = 0x000000F0;
    else if (lane_id < 12) mask = 0x00000F00;
    else if (lane_id < 16) mask = 0x0000F000;
    else if (lane_id < 20) mask = 0x000F0000;
    else if (lane_id < 24) mask = 0x00F00000;
    else if (lane_id < 28) mask = 0x0F000000;
    else                   mask = 0xF0000000;

    int lane_id_to_read_from;
    if      (lane_id <  4) lane_id_to_read_from =  0;
    else if (lane_id <  8) lane_id_to_read_from =  4;
    else if (lane_id < 12) lane_id_to_read_from =  8;
    else if (lane_id < 16) lane_id_to_read_from = 12;
    else if (lane_id < 20) lane_id_to_read_from = 16;
    else if (lane_id < 24) lane_id_to_read_from = 20;
    else if (lane_id < 28) lane_id_to_read_from = 24;
    else                   lane_id_to_read_from = 28;

    // --- Copy / MMA objects ---
    typename Kernel_traits::GmemTiledCopyQK gmem_tiled_copy_QK;
    typename Kernel_traits::GmemTiledCopyV  gmem_tiled_copy_V;
    typename Kernel_traits::GmemTiledCopyO  gmem_tiled_copy_O;

    ThrCopy thr_copy_QK = gmem_tiled_copy_QK.get_slice(threadIdx.x);
    Tensor tQgQ  = thr_copy_QK.partition_S(gQ);
    Tensor tQsQ  = thr_copy_QK.partition_D(sQ);

    // K partitions — we'll switch between cache and new at runtime
    Tensor tKsK = thr_copy_QK.partition_D(sK);
    Tensor tKrK = make_fragment_like(tKsK);

    ThrCopy thr_copy_V = gmem_tiled_copy_V.get_slice(threadIdx.x);
    Tensor tVsV = thr_copy_V.partition_D(sV);
    Tensor tVrV = make_fragment_like(tVsV);

    ThrCopy thr_copy_O = gmem_tiled_copy_O.get_slice(threadIdx.x);
    Tensor tOsO_copy = thr_copy_O.partition_S(sO);
    Tensor tOgO_copy = thr_copy_O.partition_D(gO);

    typename Kernel_traits::TiledMma tiled_mma;
    ThrMMA thr_mma_S = tiled_mma.get_slice(threadIdx.x);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);
    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);
    Tensor tSrS_float = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});

    ThrMMA thr_mma_O = tiled_mma.get_slice(threadIdx.x);
    Tensor tOsV = thr_mma_O.partition_B(sVt);
    Tensor tOrV = thr_mma_O.make_fragment_B(tOsV);
    Tensor tOrO_float = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tOsO = thr_mma_O.partition_C(sO);

    auto s2r_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{}, tiled_mma);
    auto s2r_thr_copy_Q   = s2r_tiled_copy_Q.get_slice(threadIdx.x);
    auto tSsQ_copy_view   = s2r_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view   = s2r_thr_copy_Q.retile_D(tSrQ);

    auto s2r_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomK{}, tiled_mma);
    auto s2r_thr_copy_K   = s2r_tiled_copy_K.get_slice(threadIdx.x);
    auto tSsK_copy_view   = s2r_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view   = s2r_thr_copy_K.retile_D(tSrK);

    auto s2r_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomVt{}, tiled_mma);
    auto s2r_thr_copy_V   = s2r_tiled_copy_V.get_slice(threadIdx.x);
    auto tOsV_copy_view   = s2r_thr_copy_V.partition_S(sVt);
    auto tOrV_copy_view   = s2r_thr_copy_V.retile_D(tOrV);

    // --- Block bounds ---
    // total_seqlen_k = cache_seqlen_b + seqlen_knew
    // n_block indices: 0 .. n_block_max-1
    //   blocks [0,            n_cache_blocks-1] → load from kcache/vcache
    //   blocks [n_cache_blocks, n_block_max-1]  → load from knew/vnew
    // We iterate n_block from n_block_max-1 DOWN to 0 (newest first).

    const int n_block_min = 0;
    int n_block_max = (total_seqlen_k + kBlockN - 1) / kBlockN;

    // Causal: restrict n_block_max so earlier rows don't attend to future keys
    int causal_offset   = 0;
    int is_even_mn_offset = 0;
    if constexpr (Is_causal) {
        n_block_max = max(0, (int)ceil_div((m_block + 1) * kBlockM + total_seqlen_k - seqlen_q, kBlockN));
        causal_offset = total_seqlen_k - seqlen_q - (n_block_max - 1) * kBlockN + m_block * kBlockM;
    }
    is_even_mn_offset = total_seqlen_k - (n_block_max - 1) * kBlockN;

    if (n_block_max == 0) { return; }

    int n_masking_steps = (!Is_causal)
        ? 1
        : ((Is_even_MN && Is_causal) ? ceil_div(kBlockM, kBlockN)
                                     : ceil_div(kBlockM, kBlockN) + 1);
    n_masking_steps = Is_causal ? min(n_masking_steps, n_block_max) : n_masking_steps;

    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto PV_BLOCK_MAX = size<2>(tOsV);

    clear(tOrO_float);

    // -----------------------------------------------------------------------
    // Helper lambda: load K block into tKrK register fragment.
    // n_block in [n_cache_blocks .. n_block_max-1] → knew
    // n_block in [0            .. n_cache_blocks-1] → kcache
    // -----------------------------------------------------------------------
    auto load_k_block = [&](int nb, bool do_mask) {
        if (nb >= n_cache_blocks) {
            // load from knew
            int nb_new = nb - n_cache_blocks;   // index into knew blocks
            int rows_left = seqlen_knew - nb_new * kBlockN;
            if (do_mask) {
                masked_copy<true>(gmem_tiled_copy_QK, thr_copy_QK.partition_S(gKnew)(_, _, _, nb_new),
                                  tKrK, warp_id, lane_id, rows_left, /*clear_D=*/true);
            } else {
                copy(gmem_tiled_copy_QK, thr_copy_QK.partition_S(gKnew)(_, _, _, nb_new), tKrK);
            }
        } else {
            // load from kcache
            int rows_left = cache_seqlen_b - nb * kBlockN;
            if (do_mask) {
                masked_copy<true>(gmem_tiled_copy_QK, thr_copy_QK.partition_S(gKcache)(_, _, _, nb),
                                  tKrK, warp_id, lane_id, rows_left, /*clear_D=*/true);
            } else {
                copy(gmem_tiled_copy_QK, thr_copy_QK.partition_S(gKcache)(_, _, _, nb), tKrK);
            }
        }
    };

    auto load_v_block = [&](int nb) {
        if (nb >= n_cache_blocks) {
            int nb_new = nb - n_cache_blocks;
            int rows_left = seqlen_knew - nb_new * kBlockN;
            masked_copy<true>(gmem_tiled_copy_QK, thr_copy_V.partition_S(gVnew)(_, _, _, nb_new),
                              tVsV, warp_id, lane_id, rows_left, /*clear_D=*/true);
        } else {
            int rows_left = cache_seqlen_b - nb * kBlockN;
            masked_copy<true>(gmem_tiled_copy_QK, thr_copy_V.partition_S(gVcache)(_, _, _, nb),
                              tVsV, warp_id, lane_id, rows_left, /*clear_D=*/true);
        }
    };

    // Prologue: load Q block
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QK, tQgQ, tQsQ, warp_id, lane_id,
        seqlen_q - m_block * kBlockM, /*clear_D=*/true);

    int n_block = n_block_max - 1;

    Mask<Is_causal> accum_s_mask(seqlen_q, total_seqlen_k);

    // prefetch first K block into registers
    load_k_block(n_block, /*do_mask=*/true);

    // -----------------------------------------------------------------------
    // Masking steps (boundary blocks that may need causal / padding masking)
    // -----------------------------------------------------------------------
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        copy(gmem_tiled_copy_QK, tKrK, tKsK);
        __syncthreads();
        clear(tSrS_float);

        if (n_block > n_block_min) {
            load_k_block(n_block - 1, /*do_mask=*/true);
        }

        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(s2r_tiled_copy_Q, tSsQ_copy_view(_, _, qk_block), tSrQ_copy_view(_, _, qk_block));
            copy(s2r_tiled_copy_K, tSsK_copy_view(_, _, qk_block), tSrK_copy_view(_, _, qk_block));
            gemm(tiled_mma, tSrQ(_, _, qk_block), tSrK(_, _, qk_block), tSrS_float);
        }

        __syncthreads();
        load_v_block(n_block);
        __syncthreads();

        for (int i = 0; i < tSrS_float.size(); i++) {
            tSrS_float[i] *= softmax_scale;
        }

        accum_s_mask.template apply_mask_fwd<Is_causal, Is_even_MN>(
            tSrS_float, warp_id, lane_id, kBlockN, causal_offset, is_even_mn_offset);

        // online softmax
        for (int i = 0; i < 2; i++) { rM[i] = rM_old[i]; }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < tSrS_float(make_coord(_, i), _, _).size(); j++) {
                rM[i] = fmaxf(rM[i], tSrS_float(make_coord(_, i), _, _)[j]);
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int offset = 2; offset > 0; offset /= 2) {
                rM[i] = fmaxf(rM[i], __shfl_down_sync(mask, rM[i], offset));
            }
        }
        for (int i = 0; i < 2; i++) {
            rM[i] = __shfl_sync(mask, rM[i], lane_id_to_read_from);
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < tSrS_float(make_coord(_, i), _, _).size(); j++) {
                if (rM[i] == -FLT_MAX) {
                    tSrS_float(make_coord(_, i), _, _)[j] = 0.0f;
                } else {
                    tSrS_float(make_coord(_, i), _, _)[j] =
                        expf(tSrS_float(make_coord(_, i), _, _)[j] - rM[i]);
                }
            }
        }

        for (int i = 0; i < 2; i++) {
            rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
            rD[i] = 0.0f;
        }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < tSrS_float(make_coord(_, i), _, _).size(); j++) {
                rD[i] += tSrS_float(make_coord(_, i), _, _)[j];
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int offset = 2; offset > 0; offset /= 2) {
                rD[i] += __shfl_down_sync(mask, rD[i], offset);
            }
        }
        for (int i = 0; i < 2; i++) { rL[i] += rD[i]; }
        for (int i = 0; i < 2; i++) {
            rL[i] = __shfl_sync(mask, rL[i], lane_id_to_read_from);
        }

        Tensor tOrP = convert_type<half_t>(tSrS_float);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < tOrO_float(make_coord(_, i), _, _).size(); j++) {
                tOrO_float(make_coord(_, i), _, _)[j] =
                    expf(rM_old[i] - rM[i]) * tOrO_float(make_coord(_, i), _, _)[j];
            }
        }

        CUTE_UNROLL
        for (int pv_block = 0; pv_block < PV_BLOCK_MAX; pv_block++) {
            copy(s2r_tiled_copy_V, tOsV_copy_view(_, _, pv_block), tOrV_copy_view(_, _, pv_block));
            gemm(tiled_mma, tOrP(_, _, pv_block), tOrV(_, _, pv_block), tOrO_float);
        }

        for (int i = 0; i < 2; i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Main loop (no masking needed)
    // -----------------------------------------------------------------------
    CUTE_NO_UNROLL
    for (; n_block >= n_block_min; --n_block) {
        copy(gmem_tiled_copy_QK, tKrK, tKsK);
        __syncthreads();
        clear(tSrS_float);

        if (n_block > n_block_min) {
            load_k_block(n_block - 1, /*do_mask=*/false);
        }

        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(s2r_tiled_copy_Q, tSsQ_copy_view(_, _, qk_block), tSrQ_copy_view(_, _, qk_block));
            copy(s2r_tiled_copy_K, tSsK_copy_view(_, _, qk_block), tSrK_copy_view(_, _, qk_block));
            gemm(tiled_mma, tSrQ(_, _, qk_block), tSrK(_, _, qk_block), tSrS_float);
        }

        __syncthreads();
        // For the main loop blocks they are fully within bounds; still use
        // masked_copy so we don't OOB on the last partial cache block.
        load_v_block(n_block);
        __syncthreads();

        for (int i = 0; i < tSrS_float.size(); i++) {
            tSrS_float[i] *= softmax_scale;
        }

        for (int i = 0; i < 2; i++) { rM[i] = rM_old[i]; }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < tSrS_float(make_coord(_, i), _, _).size(); j++) {
                rM[i] = fmaxf(rM[i], tSrS_float(make_coord(_, i), _, _)[j]);
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int offset = 2; offset > 0; offset /= 2) {
                rM[i] = fmaxf(rM[i], __shfl_down_sync(mask, rM[i], offset));
            }
        }
        for (int i = 0; i < 2; i++) {
            rM[i] = __shfl_sync(mask, rM[i], lane_id_to_read_from);
        }

        CUTE_UNROLL
        for (int i = 0; i < 2; i++) {
            CUTE_UNROLL
            for (int j = 0; j < tSrS_float(make_coord(_, i), _, _).size(); j++) {
                tSrS_float(make_coord(_, i), _, _)[j] =
                    expf(tSrS_float(make_coord(_, i), _, _)[j] - rM[i]);
            }
            rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
            rD[i] = 0.0f;
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < tSrS_float(make_coord(_, i), _, _).size(); j++) {
                rD[i] += tSrS_float(make_coord(_, i), _, _)[j];
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int offset = 2; offset > 0; offset /= 2) {
                rD[i] += __shfl_down_sync(mask, rD[i], offset);
            }
        }
        for (int i = 0; i < 2; i++) { rL[i] += rD[i]; }
        for (int i = 0; i < 2; i++) {
            rL[i] = __shfl_sync(mask, rL[i], lane_id_to_read_from);
        }

        Tensor tOrP = convert_type<half_t>(tSrS_float);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < tOrO_float(make_coord(_, i), _, _).size(); j++) {
                tOrO_float(make_coord(_, i), _, _)[j] =
                    expf(rM_old[i] - rM[i]) * tOrO_float(make_coord(_, i), _, _)[j];
            }
        }

        CUTE_UNROLL
        for (int pv_block = 0; pv_block < PV_BLOCK_MAX; pv_block++) {
            copy(s2r_tiled_copy_V, tOsV_copy_view(_, _, pv_block), tOrV_copy_view(_, _, pv_block));
            gemm(tiled_mma, tOrP(_, _, pv_block), tOrV(_, _, pv_block), tOrO_float);
        }

        for (int i = 0; i < 2; i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }
        __syncthreads();
    }
    // end KV loop

    // -----------------------------------------------------------------------
    // Epilogue: normalise O, write to gmem, write LSE
    // -----------------------------------------------------------------------
    for (int i = 0; i < 2; i++) {
        if (rL[i] != 0.0f) {
            for (int j = 0; j < tOrO_float(make_coord(_, i), _, _).size(); j++) {
                tOrO_float(make_coord(_, i), _, _)[j] /= rL[i];
            }
        } else {
            for (int j = 0; j < tOrO_float(make_coord(_, i), _, _).size(); j++) {
                tOrO_float(make_coord(_, i), _, _)[j] = 0.0f;
            }
        }
    }

    Tensor tOrO = convert_type<half_t>(tOrO_float);
    copy(tOrO, tOsO);
    __syncthreads();

    masked_copy<Is_even_MN>(
        gmem_tiled_copy_O, tOsO_copy, tOgO_copy, warp_id, lane_id,
        seqlen_q - m_block * kBlockM, /*clear_D=*/false);

    if (global_row_offset + thread_row < seqlen_q) {
        gL[thread_row] = (rL[0] == 0.0f) ? 0.0f : rM[0] + logf(rL[0]);
    }
    if (global_row_offset + thread_row + 8 < seqlen_q) {
        gL[thread_row + 8] = (rL[1] == 0.0f) ? 0.0f : rM[1] + logf(rL[1]);
    }
}

// ---------------------------------------------------------------------------
// Top-level device function — maps blockIdx to (bidb, bidh, m_block)
// ---------------------------------------------------------------------------
template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_attn_kvcache(
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
    const int *__restrict__ cache_seqlens,
    const int seqlen_knew,
    const int batch_size,
    const int max_seqlen_q,
    const int num_heads,
    const int num_heads_k,
    const int h_h_k_ratio,
    const int head_dim,
    const float softmax_scale)
{
    const int m_block = blockIdx.x;
    const int bidb    = blockIdx.y;
    const int bidh    = blockIdx.z;

    compute_attn_kvcache_1rowblock<Kernel_traits, Is_causal, Is_even_MN>(
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
        softmax_scale,
        bidb, bidh, m_block);
}
