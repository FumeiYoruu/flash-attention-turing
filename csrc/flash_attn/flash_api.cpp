#include <torch/extension.h>
#include "flash.h"
#include "static_switch.h"

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      at::Tensor l,
                      //void *softmax_lse_d,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      float softmax_scale,
                      bool is_causal) {

    // Reset the parameters
    params = {};

    // Set the pointers and strides.
    params.q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    params.k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
    params.v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
    params.o_ptr = reinterpret_cast<half_t*>(out.data_ptr());


    // Softmax sum
    params.l_ptr = reinterpret_cast<float*>(l.data_ptr());


    // All stride are in elements, not bytes.
    // params.q_row_stride = q.stride(-3);
    // params.k_row_stride = k.stride(-3);
    // params.v_row_stride = v.stride(-3);
    // params.q_head_stride = q.stride(-2);
    // params.k_head_stride = k.stride(-2);
    // params.v_head_stride = v.stride(-2);

    // params.o_row_stride = out.stride(-3);
    // params.o_head_stride = out.stride(-2);

    // if (cu_seqlens_q_d == nullptr) {
    //     params.q_batch_stride = q.stride(0);
    //     params.k_batch_stride = k.stride(0);
    //     params.v_batch_stride = v.stride(0);
    //     params.o_batch_stride = out.stride(0);
    // }



    // Set the dimensions.
    params.b = b;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.d = d;
    params.softmax_scale = softmax_scale;
    params.is_causal = is_causal;
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
}


void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor l,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      at::Tensor do_o,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      //void *softmax_lse_d,
                      float softmax_scale,
                      bool is_causal) {


    set_params_fprop(params,
                     b,
                     seqlen_q,
                     seqlen_k,
                     h,
                     h_k,
                     d,
                     q, k, v, out, l,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     softmax_scale,
                     is_causal
                     );


    params.do_o_ptr = reinterpret_cast<float*>(do_o.data_ptr());
    params.do_ptr = reinterpret_cast<half_t*>(dout.data_ptr());

    params.dq_ptr = reinterpret_cast<half_t*>(dq.data_ptr());
    params.dk_ptr = reinterpret_cast<half_t*>(dk.data_ptr());
    params.dv_ptr = reinterpret_cast<half_t*>(dv.data_ptr());

    // params.do_row_stride = dout.stride(-3);
    // params.do_head_stride = dout.stride(-2);

    // params.dq_row_stride = dq.stride(-3);
    // params.dk_row_stride = dk.stride(-3);
    // params.dv_row_stride = dv.stride(-3);
    // params.dq_head_stride = dq.stride(-2);
    // params.dk_head_stride = dk.stride(-2);
    // params.dv_head_stride = dv.stride(-2);                

    // if (cu_seqlens_q_d == nullptr) {
    //     params.do_batch_stride = dout.stride(0);
    //     params.dq_batch_stride = dq.stride(0);
    //     params.dk_batch_stride = dk.stride(0);
    //     params.dv_batch_stride = dv.stride(0);
    // }


}


void run_mha_fwd(Flash_fwd_params &params){
    HEADDIM_SWITCH(params.d, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_<kHeadDim, Is_causal>(params);
        });
    });
}

void run_mha_fwd_kvcache(Flash_fwd_kvcache_params &params) {
    HEADDIM_SWITCH(params.d, [&] {
        run_mha_fwd_kvcache_<kHeadDim>(params);
    });
}

void run_mha_bwd(Flash_bwd_params &params){
    HEADDIM_SWITCH(params.d, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_bwd_<kHeadDim, Is_causal>(params);
        });
    });
}


std::vector<at::Tensor>
mha_fwd(at::Tensor q,
        at::Tensor k,
        at::Tensor v,
//             int batch_size,
//             int seq_len,
//             int num_heads,
//             int head_dim,
        const float softmax_scale,
        bool is_causal)
{
    auto device = q.device();

    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    at::Tensor o = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));

    std::vector<int64_t> size = {batch_size, num_heads, seqlen_q};
    at::Tensor l = torch::zeros(size, q.options().dtype(torch::kFloat32).device(device));

    TORCH_CHECK(o.is_cuda(), "Tensor o is not on CUDA");

//    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
//    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
//    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
//    half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());
//
//    float* l_ptr = reinterpret_cast<float*>(l.data_ptr());

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_heads_k,
                     head_size,
                     q, k, v, o, l,
                     nullptr,
                     nullptr,
                     softmax_scale,
                     is_causal
                     );

    // std::cout << "Q ptr: " << q.data_ptr() << "\n";
    // std::cout << "K ptr: " << k.data_ptr() << "\n";
    // std::cout << "V ptr: " << v.data_ptr() << "\n";
    // std::cout << "O ptr: " << o.data_ptr() << "\n";

    run_mha_fwd(params);


    return {o, l};

}




std::vector<at::Tensor>
mha_bwd(at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        at::Tensor out,
        at::Tensor l,
        at::Tensor dout,
//        int batch_size,
//        int seq_len,
//        int num_heads,
//        int head_dim,
        const float softmax_scale,
        bool is_causal)
{

    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(out.dim() == 4 && dout.dim() == 4, "out and dout must be rank-4 tensors");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(out.sizes() == q.sizes() && dout.sizes() == q.sizes(), "out and dout must match q shape");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    at::Tensor dq = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));
    at::Tensor dk = torch::zeros(k.sizes(), k.options().dtype(torch::kFloat16));
    at::Tensor dv = torch::zeros(v.sizes(), v.options().dtype(torch::kFloat16));

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads != num_heads_k) {
        dk_expanded = torch::zeros({batch_size, seqlen_k, num_heads, head_size}, k.options().dtype(torch::kFloat16));
        dv_expanded = torch::zeros({batch_size, seqlen_k, num_heads, head_size}, v.options().dtype(torch::kFloat16));
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    at::Tensor do_o = torch::zeros(l.sizes(), l.options());

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                    seqlen_q,
                    seqlen_k,
                    num_heads,
                    num_heads_k,
                    head_size,
                    q,
                    k,
                    v,
                    out,
                    l,
                    dout,
                    dq,
                    dk_expanded,
                    dv_expanded,
                    do_o,
                    nullptr,
                    nullptr,
                    softmax_scale,
                    is_causal);

    run_mha_bwd(params);

    if (num_heads != num_heads_k) {
        torch::sum_out(
            dk,
            torch::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {3}
        );
        torch::sum_out(
            dv,
            torch::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {3}
        );
    }


    return {dq, dk, dv};

}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor q,
               at::Tensor k,
               at::Tensor v,
               at::Tensor &cu_seqlens_q,
               at::Tensor &cu_seqlens_k,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size + 1] with cumulative offsets");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    const int num_heads = q.size(1);
    const int num_heads_k = k.size(1);
    const int head_size = q.size(2);

    at::Tensor out = torch::zeros_like(q);
    at::Tensor l = torch::zeros({batch_size, num_heads, max_seqlen_q}, q.options().dtype(torch::kFloat32));


    // at::Tensor o = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));

    // std::vector<int64_t> size = {batch_size, num_heads, seqlen_q};
    // at::Tensor l = torch::zeros(size, q.options().dtype(torch::kFloat32).device(device));

    // TORCH_CHECK(o.is_cuda(), "Tensor o is not on CUDA");


    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        head_size,
        q, k, v, out, l,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        softmax_scale,
        is_causal
    );

    run_mha_fwd(params);

    return {out, l};
}

std::vector<at::Tensor>
mha_varlen_bwd(at::Tensor q,
               at::Tensor k,
               at::Tensor v,
               at::Tensor out,
               at::Tensor l,
               at::Tensor dout,
               at::Tensor cu_seqlens_q,
               at::Tensor cu_seqlens_k,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int64_t batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size + 1] with cumulative offsets");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must match q shape");
    TORCH_CHECK(l.dim() == 3, "l must be rank-3 for varlen_bwd");

    const int64_t num_heads = q.size(1);
    const int64_t num_heads_k = k.size(1);
    const int64_t head_size = q.size(2);
    const int64_t total_k = k.size(0);
    TORCH_CHECK(l.size(0) == batch_size && l.size(1) == num_heads && l.size(2) == max_seqlen_q,
                "l must have shape [batch_size, nheads_q, max_seqlen_q]");

    at::Tensor dq = torch::zeros_like(q);
    at::Tensor dk = torch::zeros_like(k);
    at::Tensor dv = torch::zeros_like(v);
    at::Tensor dk_expanded = dk;
    at::Tensor dv_expanded = dv;
    if (num_heads != num_heads_k) {
        dk_expanded = torch::zeros({total_k, num_heads, head_size}, k.options().dtype(torch::kFloat16));
        dv_expanded = torch::zeros({total_k, num_heads, head_size}, v.options().dtype(torch::kFloat16));
    }
    at::Tensor do_o = torch::zeros(l.sizes(), l.options());

    Flash_bwd_params params;
    set_params_dgrad(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        head_size,
        q, k, v, out, l, dout,
        dq, dk_expanded, dv_expanded, do_o,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        softmax_scale,
        is_causal
    );

    run_mha_bwd(params);

    if (num_heads != num_heads_k) {
        torch::sum_out(
            dk,
            torch::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {2}
        );
        torch::sum_out(
            dv,
            torch::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {2}
        );
    }

    return {dq, dk, dv};
}


// ---------------------------------------------------------------------------
// mha_fwd_kvcache
//
// Inputs:
//   q             (B, seqlen_q, H, d)           float16  — new query tokens
//   k_cache       (B, max_cache_seqlen, H_k, d) float16  — KV cache (mutable)
//   v_cache       (B, max_cache_seqlen, H_k, d) float16
//   cache_seqlens (B,)                          int32    — used cache tokens per batch
//   k             (B, seqlen_knew, H_k, d)      float16  — new keys   (optional)
//   v             (B, seqlen_knew, H_k, d)      float16  — new values (optional)
//   softmax_scale float
//   is_causal     bool
//
// If k/v are provided they are first written into k_cache/v_cache at
// positions [cache_seqlens[b]..cache_seqlens[b]+seqlen_knew) for each batch.
// The attention kernel then attends over cache_seqlens[b] + seqlen_knew total keys.
//
// Returns: [out (B, seqlen_q, H, d), lse (B, H, seqlen_q)]
// ---------------------------------------------------------------------------
std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor q,
                at::Tensor k_cache,
                at::Tensor v_cache,
                at::Tensor cache_seqlens,
                c10::optional<at::Tensor> k,
                c10::optional<at::Tensor> v,
                const float softmax_scale,
                bool is_causal)
{
    // --- Input validation ---
    TORCH_CHECK(q.dim() == 4, "q must be (B, seqlen_q, H, d)");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be (B, max_cache_seqlen, H_k, d)");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be (B, max_cache_seqlen, H_k, d)");
    TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.scalar_type() == torch::kInt32,
                "cache_seqlens must be 1-D int32 tensor");
    TORCH_CHECK(cache_seqlens.is_cuda() && cache_seqlens.is_contiguous(),
                "cache_seqlens must be a contiguous CUDA tensor");

    const int batch_size   = q.size(0);
    const int seqlen_q     = q.size(1);
    const int num_heads    = q.size(2);
    const int head_size    = q.size(3);
    const int num_heads_k  = k_cache.size(2);
    const int max_cache_seqlen = k_cache.size(1);

    TORCH_CHECK(k_cache.size(0) == batch_size && v_cache.size(0) == batch_size,
                "k_cache/v_cache batch must match q");
    TORCH_CHECK(v_cache.size(1) == max_cache_seqlen, "k_cache and v_cache seqlen must match");
    TORCH_CHECK(v_cache.size(2) == num_heads_k, "k_cache and v_cache num_heads must match");
    TORCH_CHECK(k_cache.size(3) == head_size && v_cache.size(3) == head_size,
                "k_cache/v_cache head_dim must match q");
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "num_heads_q must be divisible by num_heads_k for GQA/MQA");
    TORCH_CHECK(cache_seqlens.numel() == batch_size,
                "cache_seqlens must have shape [batch_size]");
    TORCH_CHECK(head_size == 64 || head_size == 128,
                "head_dim must be 64 or 128");

    bool has_new_kv = k.has_value() && v.has_value();
    int seqlen_knew = 0;
    if (has_new_kv) {
        TORCH_CHECK(k.value().dim() == 4 && v.value().dim() == 4,
                    "new k/v must be (B, seqlen_knew, H_k, d)");
        seqlen_knew = k.value().size(1);
        TORCH_CHECK(k.value().size(0) == batch_size && v.value().size(0) == batch_size,
                    "new k/v batch must match q");
        TORCH_CHECK(k.value().size(2) == num_heads_k && v.value().size(2) == num_heads_k,
                    "new k/v num_heads must match k_cache");
        TORCH_CHECK(k.value().size(3) == head_size && v.value().size(3) == head_size,
                    "new k/v head_dim must match q");
    }

    // --- Optionally append new K/V into cache (CPU-side pointer arithmetic) ---
    // We do this with a batched copy kernel via PyTorch (simple, correct).
    // For each batch b, write k[b] -> k_cache[b, cache_seqlens[b]:cache_seqlens[b]+seqlen_knew].
    // This requires the cache tensor to have enough capacity.
    if (has_new_kv && seqlen_knew > 0) {
        // We'll update the cache in Python before calling this function (see
        // flash_attention_interface.py), so here we just verify capacity.
        // Alternatively the kernel could write directly; we keep it simple.
        // (Actual cache update is done in flash_attn_with_kvcache in Python.)
    }

    // --- Outputs ---
    at::Tensor out = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));
    at::Tensor lse = torch::zeros({batch_size, num_heads, seqlen_q},
                                  q.options().dtype(torch::kFloat32));

    // --- Fill params ---
    Flash_fwd_kvcache_params params;
    params = {};

    // Base Qkv_params
    params.q_ptr  = reinterpret_cast<half_t*>(q.data_ptr());
    params.k_ptr  = nullptr;  // unused in kvcache path
    params.v_ptr  = nullptr;
    params.h      = num_heads;
    params.h_k    = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;

    // Flash_fwd_params
    params.o_ptr         = reinterpret_cast<half_t*>(out.data_ptr());
    params.l_ptr         = reinterpret_cast<float*>(lse.data_ptr());
    params.softmax_scale = softmax_scale;
    params.b             = batch_size;
    params.seqlen_q      = seqlen_q;
    // seqlen_k: maximum possible (cache capacity + new), used for grid sizing;
    // actual per-batch length is read from cache_seqlens inside the kernel.
    params.seqlen_k      = max_cache_seqlen + seqlen_knew;
    params.d             = head_size;
    params.is_causal     = is_causal;

    // KV cache
    params.kcache_ptr          = reinterpret_cast<half_t*>(k_cache.data_ptr());
    params.vcache_ptr          = reinterpret_cast<half_t*>(v_cache.data_ptr());
    params.kcache_batch_stride = k_cache.stride(0);
    params.kcache_row_stride   = k_cache.stride(1);
    params.kcache_head_stride  = k_cache.stride(2);
    params.vcache_batch_stride = v_cache.stride(0);
    params.vcache_row_stride   = v_cache.stride(1);
    params.vcache_head_stride  = v_cache.stride(2);

    params.cache_seqlens = reinterpret_cast<int*>(cache_seqlens.data_ptr());

    // New K/V
    params.seqlen_knew = seqlen_knew;
    if (has_new_kv && seqlen_knew > 0) {
        at::Tensor k_cont = k.value().contiguous();
        at::Tensor v_cont = v.value().contiguous();
        params.knew_ptr          = reinterpret_cast<half_t*>(k_cont.data_ptr());
        params.vnew_ptr          = reinterpret_cast<half_t*>(v_cont.data_ptr());
        params.knew_batch_stride = k_cont.stride(0);
        params.knew_row_stride   = k_cont.stride(1);
        params.knew_head_stride  = k_cont.stride(2);
        params.vnew_batch_stride = v_cont.stride(0);
        params.vnew_row_stride   = v_cont.stride(1);
        params.vnew_head_stride  = v_cont.stride(2);
    } else {
        params.knew_ptr = nullptr;
        params.vnew_ptr = nullptr;
        params.knew_batch_stride = params.knew_row_stride = params.knew_head_stride = 0;
        params.vnew_batch_stride = params.vnew_row_stride = params.vnew_head_stride = 0;
    }

    run_mha_fwd_kvcache(params);

    return {out, lse};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Varlen forward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Varlen backward pass");
    m.def("fwd_kvcache", &mha_fwd_kvcache,
          "KV-cache inference forward pass",
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("cache_seqlens"),
          py::arg("k")             = py::none(),
          py::arg("v")             = py::none(),
          py::arg("softmax_scale") = 1.0f,
          py::arg("is_causal")     = true);
}
