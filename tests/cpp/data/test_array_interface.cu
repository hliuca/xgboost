/*!
 * Copyright 2021 by Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include "../helpers.h"
#include "../../../src/data/array_interface.h"

namespace xgboost {

__global__ void SleepForTest(uint64_t *out, uint64_t duration) {
  auto start = clock64();
  auto t = 0;
  while (t < duration) {
    t = clock64() - start;
  }
  out[0] = t;
}

TEST(ArrayInterface, Stream) {
  size_t constexpr kRows = 10, kCols = 10;
  HostDeviceVector<float> storage;
  auto arr_str = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);

#if defined(XGBOOST_USE_CUDA)
  cudaStream_t stream;
  cudaStreamCreate(&stream);
#elif defined(XGBOOST_USE_HIP)
  hipStream_t stream;
  hipStreamCreate(&stream);
#endif

  auto j_arr =Json::Load(StringView{arr_str});
  j_arr["stream"] = Integer(reinterpret_cast<int64_t>(stream));
  Json::Dump(j_arr, &arr_str);

  dh::caching_device_vector<uint64_t> out(1, 0);
  uint64_t dur = 1e9;
  dh::LaunchKernel{1, 1, 0, stream}(SleepForTest, out.data().get(), dur);
  ArrayInterface<2> arr(arr_str);

  auto t = out[0];
  CHECK_GE(t, dur);

#if defined(XGBOOST_USE_CUDA)
  cudaStreamDestroy(stream);
#elif defined(XGBOOST_USE_HIP)
  hipStreamDestroy(stream);
#endif
}

TEST(ArrayInterface, Ptr) {
  std::vector<float> h_data(10);
  ASSERT_FALSE(ArrayInterfaceHandler::IsCudaPtr(h_data.data()));
#if defined(XGBOOST_USE_CUDA)
  dh::safe_cuda(cudaGetLastError());
#elif defined(XGBOOST_USE_HIP)
  dh::safe_cuda(hipGetLastError());
#endif

  dh::device_vector<float> d_data(10);
  ASSERT_TRUE(ArrayInterfaceHandler::IsCudaPtr(d_data.data().get()));
#if defined(XGBOOST_USE_CUDA)
  dh::safe_cuda(cudaGetLastError());
#elif defined(XGBOOST_USE_HIP)
  dh::safe_cuda(hipGetLastError());
#endif

  ASSERT_FALSE(ArrayInterfaceHandler::IsCudaPtr(nullptr));
#if defined(XGBOOST_USE_CUDA)
  dh::safe_cuda(cudaGetLastError());
#elif defined(XGBOOST_USE_HIP)
  dh::safe_cuda(hipGetLastError());
#endif
}
}  // namespace xgboost
