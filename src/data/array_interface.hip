/*!
 * Copyright 2021 by Contributors
 */
#include "../common/common.h"
#include "array_interface.h"

namespace xgboost {
void ArrayInterfaceHandler::SyncCudaStream(int64_t stream) {
  switch (stream) {
    case 0:
      /**
       * disallowed by the `__cuda_array_interface__`.  Quote:
       *
       *   This is disallowed as it would be ambiguous between None and the default
       *   stream, and also between the legacy and per-thread default streams. Any use
       *   case where 0 might be given should either use None, 1, or 2 instead for
       *   clarity.
       */
      LOG(FATAL) << "Invalid stream ID in array interface: " << stream;
    case 1:
      // default legacy stream
      break;
    case 2:
      // default per-thread stream
    default:
#if defined(XGBOOST_USE_CUDA)
      dh::safe_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
#elif defined(XGBOOST_USE_HIP)
      dh::safe_cuda(hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream)));
#endif
  }
}

bool ArrayInterfaceHandler::IsCudaPtr(void const* ptr) {
  if (!ptr) {
    return false;
  }

#if defined(XGBOOST_USE_CUDA)
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, ptr);
  // reset error
  CHECK_EQ(err, cudaGetLastError());
  if (err == cudaErrorInvalidValue) {
    // CUDA < 11
    return false;
  } else if (err == cudaSuccess) {
    // CUDA >= 11
    switch (attr.type) {
      case cudaMemoryTypeUnregistered:
      case cudaMemoryTypeHost:
        return false;
      default:
        return true;
    }
    return true;
  } else {
    // other errors, `cudaErrorNoDevice`, `cudaErrorInsufficientDriver` etc.
    return false;
  }
#elif defined(XGBOOST_USE_HIP)
  hipPointerAttribute_t attr;
  auto err = hipPointerGetAttributes(&attr, ptr);
  // reset error
  CHECK_EQ(err, hipGetLastError());
  if (err == hipErrorInvalidValue) {
    return false;
  } else if (err == hipSuccess) {
    switch (attr.memoryType) {
      case hipMemoryTypeUnified:
      case hipMemoryTypeHost:
        return false;
      default:
        return true;
    }
    return true;
  } else {
    return false;
  }
#endif
}
}  // namespace xgboost
