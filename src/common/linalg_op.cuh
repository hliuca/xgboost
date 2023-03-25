/*!
 * Copyright 2021-2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_LINALG_OP_CUH_
#define XGBOOST_COMMON_LINALG_OP_CUH_

#include "device_helpers.cuh"
#include "linalg_op.h"
#include "xgboost/context.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace linalg {
template <typename T, int32_t D, typename Fn>
#if defined(XGBOOST_USE_HIP)
void ElementWiseKernelDevice(linalg::TensorView<T, D> t, Fn&& fn, hipStream_t s = nullptr)
#elif defined(XGBOOST_USE_CUDA)
void ElementWiseKernelDevice(linalg::TensorView<T, D> t, Fn&& fn, cudaStream_t s = nullptr)
#endif
{
#if defined(XGBOOST_USE_HIP)
  dh::safe_cuda(hipSetDevice(t.DeviceIdx()));
#elif defined(XGBOOST_USE_CUDA)
  dh::safe_cuda(cudaSetDevice(t.DeviceIdx()));
#endif

  static_assert(std::is_void<std::result_of_t<Fn(size_t, T&)>>::value,
                "For function with return, use transform instead.");
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) mutable { fn(i, ptr[i]); });
  } else {
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) mutable {
      T& v = detail::Apply(t, linalg::UnravelIndex(i, t.Shape()));
      fn(i, v);
    });
  }
}

template <typename T, int32_t D, typename Fn>
#if defined(XGBOOST_USE_HIP)
void ElementWiseTransformDevice(linalg::TensorView<T, D> t, Fn&& fn, hipStream_t s = nullptr)
#elif defined(XGBOOST_USE_CUDA)
void ElementWiseTransformDevice(linalg::TensorView<T, D> t, Fn&& fn, cudaStream_t s = nullptr)
#endif
{
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) { ptr[i] = fn(i, ptr[i]); });
  } else {
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) mutable {
      T& v = detail::Apply(t, linalg::UnravelIndex(i, t.Shape()));
      v = fn(i, v);
    });
  }
}

template <typename T, int32_t D, typename Fn>
void ElementWiseKernel(Context const* ctx, linalg::TensorView<T, D> t, Fn&& fn) {
  ctx->IsCPU() ? ElementWiseKernelHost(t, ctx->Threads(), fn) : ElementWiseKernelDevice(t, fn);
}
}  // namespace linalg
}  // namespace xgboost
#endif  // XGBOOST_COMMON_LINALG_OP_CUH_
