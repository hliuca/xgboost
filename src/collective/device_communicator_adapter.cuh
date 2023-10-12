/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once

#include "communicator.h"
#include "device_communicator.cuh"

namespace xgboost {
namespace collective {

class DeviceCommunicatorAdapter : public DeviceCommunicator {
 public:
  explicit DeviceCommunicatorAdapter(int device_ordinal)
      : device_ordinal_{device_ordinal}, world_size_{GetWorldSize()}, rank_{GetRank()} {
    if (device_ordinal_ < 0) {
      LOG(FATAL) << "Invalid device ordinal: " << device_ordinal_;
    }
  }

  ~DeviceCommunicatorAdapter() override = default;

  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override {
    if (world_size_ == 1) {
      return;
    }

#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipSetDevice(device_ordinal_));
#endif
    auto size = count * GetTypeSize(data_type);
    host_buffer_.resize(size);
#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaMemcpy(host_buffer_.data(), send_receive_buffer, size, cudaMemcpyDefault));
    Allreduce(host_buffer_.data(), count, data_type, op);
    dh::safe_cuda(cudaMemcpy(send_receive_buffer, host_buffer_.data(), size, cudaMemcpyDefault));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipMemcpy(host_buffer_.data(), send_receive_buffer, size, hipMemcpyDefault));
    AllReduce(host_buffer_.data(), count, data_type, op);
    dh::safe_cuda(hipMemcpy(send_receive_buffer, host_buffer_.data(), size, hipMemcpyDefault));
#endif
  }

  void AllGather(void const *send_buffer, void *receive_buffer, std::size_t send_size) override {
    if (world_size_ == 1) {
      return;
    }

#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    host_buffer_.resize(send_size * world_size_);
    dh::safe_cuda(cudaMemcpy(host_buffer_.data() + rank_ * send_size, send_buffer, send_size,
                             cudaMemcpyDefault));
    Allgather(host_buffer_.data(), host_buffer_.size());
    dh::safe_cuda(
        cudaMemcpy(receive_buffer, host_buffer_.data(), host_buffer_.size(), cudaMemcpyDefault));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipSetDevice(device_ordinal_));
    host_buffer_.resize(send_size * world_size_);
    dh::safe_cuda(hipMemcpy(host_buffer_.data() + rank_ * send_size, send_buffer, send_size,
                             hipMemcpyDefault));
    Allgather(host_buffer_.data(), host_buffer_.size());
    dh::safe_cuda(
        hipMemcpy(receive_buffer, host_buffer_.data(), host_buffer_.size(), hipMemcpyDefault));
#endif
  }

  void AllGatherV(void const *send_buffer, size_t length_bytes, std::vector<std::size_t> *segments,
                  dh::caching_device_vector<char> *receive_buffer) override {
    if (world_size_ == 1) {
      return;
    }

#if defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipSetDevice(device_ordinal_));
#elif defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
#endif

    segments->clear();
    segments->resize(world_size_, 0);
    segments->at(rank_) = length_bytes;
    Allreduce(segments->data(), segments->size(), DataType::kUInt64, Operation::kMax);
    auto total_bytes = std::accumulate(segments->cbegin(), segments->cend(), 0UL);
    receive_buffer->resize(total_bytes);

    host_buffer_.resize(total_bytes);
    size_t offset = 0;
    for (int32_t i = 0; i < world_size_; ++i) {
      size_t as_bytes = segments->at(i);
      if (i == rank_) {
#if defined(XGBOOST_USE_CUDA)
        dh::safe_cuda(cudaMemcpy(host_buffer_.data() + offset, send_buffer, segments->at(rank_),
                                 cudaMemcpyDefault));
#elif defined(XGBOOST_USE_HIP)
        dh::safe_cuda(hipMemcpy(host_buffer_.data() + offset, send_buffer, segments->at(rank_),
                                 hipMemcpyDefault));
#endif
      }
      Broadcast(host_buffer_.data() + offset, as_bytes, i);
      offset += as_bytes;
    }

#if defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipMemcpy(receive_buffer->data().get(), host_buffer_.data(), total_bytes,
                             hipMemcpyDefault));
#elif defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaMemcpy(receive_buffer->data().get(), host_buffer_.data(), total_bytes,
                             cudaMemcpyDefault));
#endif
  }

  void Synchronize() override {
    // Noop.
  }

 private:
  int const device_ordinal_;
  int const world_size_;
  int const rank_;
  /// Host buffer used to call communicator functions.
  std::vector<char> host_buffer_{};
};

}  // namespace collective
}  // namespace xgboost
