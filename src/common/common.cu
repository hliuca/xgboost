/*!
 * Copyright 2018-2022 XGBoost contributors
 */
#include "common.h"

namespace xgboost {
namespace common {

void SetDevice(std::int32_t device) {
  if (device >= 0) {
#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaSetDevice(device));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipSetDevice(device));
#endif
  }
}

int AllVisibleGPUs() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDeviceCount will fail.
#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipGetDeviceCount(&n_visgpus));
#endif
  } catch (const dmlc::Error &) {
#if defined(XGBOOST_USE_CUDA)
    cudaGetLastError();  // reset error.
#elif defined(XGBOOST_USE_HIP)
    hipGetLastError();  // reset error.
#endif
    return 0;
  }
  return n_visgpus;
}

}  // namespace common
}  // namespace xgboost
