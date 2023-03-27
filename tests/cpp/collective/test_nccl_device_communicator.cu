/**
 * Copyright 2022-2023, XGBoost contributors
 */
#if defined(XGBOOST_USE_NCCL) ||defined(XGBOOST_USE_RCCL)

#include <gtest/gtest.h>

#include <string>  // for string

#if defined(XGBOOST_USE_NCCL)
#include "../../../src/collective/nccl_device_communicator.cuh"
#elif defined(XGBOOST_USE_RCCL)
#include "../../../src/collective/nccl_device_communicator.hip.h"
#endif

namespace xgboost {
namespace collective {

TEST(NcclDeviceCommunicatorSimpleTest, ThrowOnInvalidDeviceOrdinal) {
  auto construct = []() { NcclDeviceCommunicator comm{-1, nullptr}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(NcclDeviceCommunicatorSimpleTest, ThrowOnInvalidCommunicator) {
  auto construct = []() { NcclDeviceCommunicator comm{0, nullptr}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(NcclDeviceCommunicatorSimpleTest, SystemError) {
  try {
    dh::safe_nccl(ncclSystemError);
  } catch (dmlc::Error const& e) {
    auto str = std::string{e.what()};
    ASSERT_TRUE(str.find("environment variables") != std::string::npos);
  }
}
}  // namespace collective
}  // namespace xgboost

#endif  // XGBOOST_USE_NCCL || XGBOOST_USE_RCCL
