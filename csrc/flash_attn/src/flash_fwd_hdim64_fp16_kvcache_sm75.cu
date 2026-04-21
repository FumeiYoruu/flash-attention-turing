#include "flash_fwd_kvcache_launch_template.h"
#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

template<>
void run_mha_fwd_kvcache_<64>(Flash_fwd_kvcache_params &params) {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_mha_fwd_kvcache_hdim64<Is_causal>(params);
    });
}
