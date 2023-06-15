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
  DeviceCommunicatorAdapter(int device_ordinal, Communicator *communicator)
      : device_ordinal_{device_ordinal}, communicator_{communicator} {
    if (device_ordinal_ < 0) {
      LOG(FATAL) << "Invalid device ordinal: " << device_ordinal_;
    }
    if (communicator_ == nullptr) {
      LOG(FATAL) << "Communicator cannot be null.";
    }
  }

  ~DeviceCommunicatorAdapter() override = default;

  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override {
    if (communicator_->GetWorldSize() == 1) {
      return;
    }

#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipSetDevice(device_ordinal_));
#endif
    auto size = count * GetTypeSize(data_type);
    host_buffer_.reserve(size);
#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaMemcpy(host_buffer_.data(), send_receive_buffer, size, cudaMemcpyDefault));
    communicator_->AllReduce(host_buffer_.data(), count, data_type, op);
    dh::safe_cuda(cudaMemcpy(send_receive_buffer, host_buffer_.data(), size, cudaMemcpyDefault));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipMemcpy(host_buffer_.data(), send_receive_buffer, size, hipMemcpyDefault));
    communicator_->AllReduce(host_buffer_.data(), count, data_type, op);
    dh::safe_cuda(hipMemcpy(send_receive_buffer, host_buffer_.data(), size, hipMemcpyDefault));
#endif
  }

  void AllGatherV(void const *send_buffer, size_t length_bytes, std::vector<std::size_t> *segments,
                  dh::caching_device_vector<char> *receive_buffer) override {
    if (communicator_->GetWorldSize() == 1) {
      return;
    }

#if defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipSetDevice(device_ordinal_));
#else
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
#endif

    int const world_size = communicator_->GetWorldSize();
    int const rank = communicator_->GetRank();

    segments->clear();
    segments->resize(world_size, 0);
    segments->at(rank) = length_bytes;
    communicator_->AllReduce(segments->data(), segments->size(), DataType::kUInt64,
                             Operation::kMax);
    auto total_bytes = std::accumulate(segments->cbegin(), segments->cend(), 0UL);
    receive_buffer->resize(total_bytes);

    host_buffer_.reserve(total_bytes);
    size_t offset = 0;
    for (int32_t i = 0; i < world_size; ++i) {
      size_t as_bytes = segments->at(i);
      if (i == rank) {
#if defined(XGBOOST_USE_HIP)
        dh::safe_cuda(hipMemcpy(host_buffer_.data() + offset, send_buffer, segments->at(rank),
                                 hipMemcpyDefault));
#else
        dh::safe_cuda(cudaMemcpy(host_buffer_.data() + offset, send_buffer, segments->at(rank),
                                 cudaMemcpyDefault));
#endif
      }
      communicator_->Broadcast(host_buffer_.data() + offset, as_bytes, i);
      offset += as_bytes;
    }

#if defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipMemcpy(receive_buffer->data().get(), host_buffer_.data(), total_bytes,
                             hipMemcpyDefault));
#else
    dh::safe_cuda(cudaMemcpy(receive_buffer->data().get(), host_buffer_.data(), total_bytes,
                             cudaMemcpyDefault));
#endif
  }

  void Synchronize() override {
    // Noop.
  }

 private:
  int const device_ordinal_;
  Communicator *communicator_;
  /// Host buffer used to call communicator functions.
  std::vector<char> host_buffer_{};
};

}  // namespace collective
}  // namespace xgboost
