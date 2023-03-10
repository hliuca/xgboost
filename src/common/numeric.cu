/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <thrust/execution_policy.h>

#if defined(XGBOOST_USE_CUDA)
#include "device_helpers.cuh"            // dh::Reduce, dh::XGBCachingDeviceAllocator
#elif defined(XGBOOST_USE_HIP)
#include "device_helpers.hip.h"          // dh::Reduce, dh::XGBCachingDeviceAllocator
#endif

#include "numeric.h"
#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace common {
namespace cuda_impl {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  values.SetDevice(ctx->gpu_id);
  auto const d_values = values.ConstDeviceSpan();
  dh::XGBCachingDeviceAllocator<char> alloc;

#if defined(XGBOOST_USE_CUDA)
  return dh::Reduce(thrust::cuda::par(alloc), dh::tcbegin(d_values), dh::tcend(d_values), 0.0,
                    thrust::plus<float>{});
#elif defined(XGBOOST_USE_HIP)
  return dh::Reduce(thrust::hip::par(alloc), dh::tcbegin(d_values), dh::tcend(d_values), 0.0,
                    thrust::plus<float>{});
#endif
}
}  // namespace cuda_impl
}  // namespace common
}  // namespace xgboost
