/**
 * Copyright 2019-2023, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>
#include "../../../src/common/bitfield.h"
#if defined(XGBOOST_USE_CUDA)
#include "../../../src/common/device_helpers.cuh"
#elif defined(XGBOOST_USE_HIP)
#include "../../../src/common/device_helpers.hip.h"
#endif

namespace xgboost {

__global__ void TestSetKernel(LBitField64 bits) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < bits.Capacity()) {
    bits.Set(tid);
  }
}

TEST(BitField, StorageSize) {
  size_t constexpr kElements { 16 };
  size_t size = LBitField64::ComputeStorageSize(kElements);
  ASSERT_EQ(1, size);
  size = RBitField8::ComputeStorageSize(4);
  ASSERT_EQ(1, size);
  size = RBitField8::ComputeStorageSize(kElements);
  ASSERT_EQ(2, size);
}

TEST(BitField, GPUSet) {
  dh::device_vector<LBitField64::value_type> storage;
  uint32_t constexpr kBits = 128;
  storage.resize(128);
  auto bits = LBitField64(dh::ToSpan(storage));
  TestSetKernel<<<1, kBits>>>(bits);

  std::vector<LBitField64::value_type> h_storage(storage.size());
  thrust::copy(storage.begin(), storage.end(), h_storage.begin());
  LBitField64 outputs{
      common::Span<LBitField64::value_type>{h_storage.data(), h_storage.data() + h_storage.size()}};
  for (size_t i = 0; i < kBits; ++i) {
    ASSERT_TRUE(outputs.Check(i));
  }
}

namespace {
template <bool is_and, typename Op>
void TestGPULogic(Op op) {
  uint32_t constexpr kBits = 128;
  dh::device_vector<LBitField64::value_type> lhs_storage(kBits);
  dh::device_vector<LBitField64::value_type> rhs_storage(kBits);
  auto lhs = LBitField64(dh::ToSpan(lhs_storage));
  auto rhs = LBitField64(dh::ToSpan(rhs_storage));
  thrust::fill(lhs_storage.begin(), lhs_storage.end(), 0UL);
  thrust::fill(rhs_storage.begin(), rhs_storage.end(), ~static_cast<LBitField64::value_type>(0UL));
  dh::LaunchN(kBits, [=] __device__(auto) mutable { op(lhs, rhs); });

  std::vector<LBitField64::value_type> h_storage(lhs_storage.size());
  thrust::copy(lhs_storage.begin(), lhs_storage.end(), h_storage.begin());
  LBitField64 outputs{{h_storage.data(), h_storage.data() + h_storage.size()}};
  if (is_and) {
    for (size_t i = 0; i < kBits; ++i) {
      ASSERT_FALSE(outputs.Check(i));
    }
  } else {
    for (size_t i = 0; i < kBits; ++i) {
      ASSERT_TRUE(outputs.Check(i));
    }
  }
}

void TestGPUAnd() {
  TestGPULogic<true>([] XGBOOST_DEVICE(LBitField64 & lhs, LBitField64 const& rhs) { lhs &= rhs; });
}

void TestGPUOr() {
  TestGPULogic<false>([] XGBOOST_DEVICE(LBitField64 & lhs, LBitField64 const& rhs) { lhs |= rhs; });
}
}  // namespace

TEST(BitField, GPUAnd) { TestGPUAnd(); }

TEST(BitField, GPUOr) { TestGPUOr(); }
}  // namespace xgboost
