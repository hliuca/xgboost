/**
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_CUDA_CONTEXT_CUH_
#define XGBOOST_COMMON_CUDA_CONTEXT_CUH_
#include <thrust/execution_policy.h>

#include "device_helpers.cuh"

namespace xgboost {
struct CUDAContext {
 private:
  dh::XGBCachingDeviceAllocator<char> caching_alloc_;
  dh::XGBDeviceAllocator<char> alloc_;

 public:
  /**
   * \brief Caching thrust policy.
   */
#if defined(XGBOOST_USE_HIP)
  auto CTP() const { return thrust::hip::par(caching_alloc_).on(dh::DefaultStream()); }
#else
  auto CTP() const { return thrust::cuda::par(caching_alloc_).on(dh::DefaultStream()); }
#endif

  /**
   * \brief Thrust policy without caching allocator.
   */
#if defined(XGBOOST_USE_HIP)
  auto TP() const { return thrust::hip::par(alloc_).on(dh::DefaultStream()); }
#else
  auto TP() const { return thrust::cuda::par(alloc_).on(dh::DefaultStream()); }
#endif

  auto Stream() const { return dh::DefaultStream(); }
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_CUDA_CONTEXT_CUH_
