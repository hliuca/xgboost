/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include <dmlc/registry.h>
#include <thrust/iterator/counting_iterator.h>  // make_counting_iterator
#include <thrust/reduce.h>                      // reduce
#include <xgboost/metric.h>

#include <cstddef>                       // std::size_t
#include <memory>                        // std::shared_ptr

#include "../common/cuda_context.cuh"    // CUDAContext
#include "metric_common.h"
#include "xgboost/base.h"                // XGBOOST_DEVICE
#include "xgboost/context.h"             // Context
#include "xgboost/data.h"                // MetaInfo
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(rank_metric_gpu);

/*! \brief Evaluate rank list on GPU */
template <typename EvalMetricT>
struct EvalRankGpu : public GPUMetric, public EvalRankConfig {
 public:
  double Eval(const HostDeviceVector<bst_float> &preds, const MetaInfo &info) override {
    // Sanity check is done by the caller
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(preds.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;

    const auto ngroups = static_cast<bst_omp_uint>(gptr.size() - 1);

    auto device = ctx_->gpu_id;

#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaSetDevice(device));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipSetDevice(device));
#endif

    info.labels.SetDevice(device);
    preds.SetDevice(device);

    auto dpreds = preds.ConstDevicePointer();
    auto dlabels = info.labels.View(device);

    // Sort all the predictions
    dh::SegmentSorter<float> segment_pred_sorter;
    segment_pred_sorter.SortItems(dpreds, preds.Size(), gptr);

    // Compute individual group metric and sum them up
    return EvalMetricT::EvalMetric(segment_pred_sorter, dlabels.Values().data(), *this);
  }

  const char* Name() const override {
    return name.c_str();
  }

  explicit EvalRankGpu(const char* name, const char* param) {
    using namespace std;  // NOLINT(*)
    if (param != nullptr) {
      std::ostringstream os;
      if (sscanf(param, "%u[-]?", &this->topn) == 1) {
        os << name << '@' << param;
        this->name = os.str();
      } else {
        os << name << param;
        this->name = os.str();
      }
      if (param[strlen(param) - 1] == '-') {
        this->minus = true;
      }
    } else {
      this->name = name;
    }
  }
};

/*! \brief Precision at N, for both classification and rank */
struct EvalPrecisionGpu {
 public:
  static double EvalMetric(const dh::SegmentSorter<float> &pred_sorter,
                           const float *dlabels,
                           const EvalRankConfig &ecfg) {
    // Group info on device
    const auto &dgroups = pred_sorter.GetGroupsSpan();
    const auto ngroups = pred_sorter.GetNumGroups();
    const auto &dgroup_idx = pred_sorter.GetGroupSegmentsSpan();

    // Original positions of the predictions after they have been sorted
    const auto &dpreds_orig_pos = pred_sorter.GetOriginalPositionsSpan();

    // First, determine non zero labels in the dataset individually
    auto DetermineNonTrivialLabelLambda = [=] __device__(uint32_t idx) {
      return (static_cast<unsigned>(dlabels[dpreds_orig_pos[idx]]) != 0) ? 1 : 0;
    };  // NOLINT

    // Find each group's metric sum
    dh::caching_device_vector<uint32_t> hits(ngroups, 0);
    const auto nitems = pred_sorter.GetNumItems();
    auto *dhits = hits.data().get();

    int device_id = -1;

#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaGetDevice(&device_id));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipGetDevice(&device_id));
#endif

    // For each group item compute the aggregated precision
    dh::LaunchN(nitems, nullptr, [=] __device__(uint32_t idx) {
      const auto group_idx = dgroup_idx[idx];
      const auto group_begin = dgroups[group_idx];
      const auto ridx = idx - group_begin;
      if (ridx < ecfg.topn && DetermineNonTrivialLabelLambda(idx)) {
        atomicAdd(&dhits[group_idx], 1);
      }
    });

    // Allocator to be used for managing space overhead while performing reductions
    dh::XGBCachingDeviceAllocator<char> alloc;

#if defined(XGBOOST_USE_CUDA)
    return static_cast<double>(thrust::reduce(thrust::cuda::par(alloc),
                                              hits.begin(), hits.end())) / ecfg.topn;
#elif defined(XGBOOST_USE_HIP)
    return static_cast<double>(thrust::reduce(thrust::hip::par(alloc),
                                              hits.begin(), hits.end())) / ecfg.topn;
#endif
  }
};

/*! \brief NDCG: Normalized Discounted Cumulative Gain at N */
struct EvalNDCGGpu {
 public:
  static void ComputeDCG(const dh::SegmentSorter<float> &pred_sorter,
                         const float *dlabels,
                         const EvalRankConfig &ecfg,
                         // The order in which labels have to be accessed. The order is determined
                         // by sorting the predictions or the labels for the entire dataset
                         const xgboost::common::Span<const uint32_t> &dlabels_sort_order,
                         dh::caching_device_vector<double> *dcgptr) {
    dh::caching_device_vector<double> &dcgs(*dcgptr);
    // Group info on device
    const auto &dgroups = pred_sorter.GetGroupsSpan();
    const auto &dgroup_idx = pred_sorter.GetGroupSegmentsSpan();

    // First, determine non zero labels in the dataset individually
    auto DetermineNonTrivialLabelLambda = [=] __device__(uint32_t idx) {
      return (static_cast<unsigned>(dlabels[dlabels_sort_order[idx]]));
    };  // NOLINT

    // Find each group's DCG value
    const auto nitems = pred_sorter.GetNumItems();
    auto *ddcgs = dcgs.data().get();

    int device_id = -1;

#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaGetDevice(&device_id));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipGetDevice(&device_id));
#endif

    // For each group item compute the aggregated precision
    dh::LaunchN(nitems, nullptr, [=] __device__(uint32_t idx) {
      const auto group_idx = dgroup_idx[idx];
      const auto group_begin = dgroups[group_idx];
      const auto ridx = idx - group_begin;
      auto label = DetermineNonTrivialLabelLambda(idx);
      if (ridx < ecfg.topn && label) {
        atomicAdd(&ddcgs[group_idx], ((1 << label) - 1) / std::log2(ridx + 2.0));
      }
    });
  }

  static double EvalMetric(const dh::SegmentSorter<float> &pred_sorter,
                           const float *dlabels,
                           const EvalRankConfig &ecfg) {
    // Sort the labels and compute IDCG
    dh::SegmentSorter<float> segment_label_sorter;
    segment_label_sorter.SortItems(dlabels, pred_sorter.GetNumItems(),
                                   pred_sorter.GetGroupSegmentsSpan());

    uint32_t ngroups = pred_sorter.GetNumGroups();

    dh::caching_device_vector<double> idcg(ngroups, 0);
    ComputeDCG(pred_sorter, dlabels, ecfg, segment_label_sorter.GetOriginalPositionsSpan(), &idcg);

    // Compute the DCG values next
    dh::caching_device_vector<double> dcg(ngroups, 0);
    ComputeDCG(pred_sorter, dlabels, ecfg, pred_sorter.GetOriginalPositionsSpan(), &dcg);

    double *ddcg = dcg.data().get();
    double *didcg = idcg.data().get();

    int device_id = -1;

#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaGetDevice(&device_id));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipGetDevice(&device_id));
#endif

    // Compute the group's DCG and reduce it across all groups
    dh::LaunchN(ngroups, nullptr, [=] __device__(uint32_t gidx) {
      if (didcg[gidx] == 0.0f) {
        ddcg[gidx] = (ecfg.minus) ? 0.0f : 1.0f;
      } else {
        ddcg[gidx] /= didcg[gidx];
      }
    });

    // Allocator to be used for managing space overhead while performing reductions
    dh::XGBCachingDeviceAllocator<char> alloc;

#if defined(XGBOOST_USE_CUDA)
    return thrust::reduce(thrust::cuda::par(alloc), dcg.begin(), dcg.end());
#elif defined(XGBOOST_USE_HIP)
    return thrust::reduce(thrust::hip::par(alloc), dcg.begin(), dcg.end());
#endif
  }
};

/*! \brief Mean Average Precision at N, for both classification and rank */
struct EvalMAPGpu {
 public:
  static double EvalMetric(const dh::SegmentSorter<float> &pred_sorter,
                           const float *dlabels,
                           const EvalRankConfig &ecfg) {
    // Group info on device
    const auto &dgroups = pred_sorter.GetGroupsSpan();
    const auto ngroups = pred_sorter.GetNumGroups();
    const auto &dgroup_idx = pred_sorter.GetGroupSegmentsSpan();

    // Original positions of the predictions after they have been sorted
    const auto &dpreds_orig_pos = pred_sorter.GetOriginalPositionsSpan();

    // First, determine non zero labels in the dataset individually
    const auto nitems = pred_sorter.GetNumItems();
    dh::caching_device_vector<uint32_t> hits(nitems, 0);
    auto DetermineNonTrivialLabelLambda = [=] __device__(uint32_t idx) {
      return (static_cast<unsigned>(dlabels[dpreds_orig_pos[idx]]) != 0) ? 1 : 0;
    };  // NOLINT

    thrust::transform(thrust::make_counting_iterator(static_cast<uint32_t>(0)),
                      thrust::make_counting_iterator(nitems),
                      hits.begin(),
                      DetermineNonTrivialLabelLambda);

    // Allocator to be used by sort for managing space overhead while performing prefix scans
    dh::XGBCachingDeviceAllocator<char> alloc;

    // Next, prefix scan the nontrivial labels that are segmented to accumulate them.
    // This is required for computing the metric sum
    // Data segmented into different groups...
#if defined(XGBOOST_USE_CUDA)
    thrust::inclusive_scan_by_key(thrust::cuda::par(alloc),
                                  dh::tcbegin(dgroup_idx), dh::tcend(dgroup_idx),
                                  hits.begin(),  // Input value
                                  hits.begin());  // In-place scan
#elif defined(XGBOOST_USE_HIP)
    thrust::inclusive_scan_by_key(thrust::hip::par(alloc),
                                  dh::tcbegin(dgroup_idx), dh::tcend(dgroup_idx),
                                  hits.begin(),  // Input value
                                  hits.begin());  // In-place scan
#endif

    // Find each group's metric sum
    dh::caching_device_vector<double> sumap(ngroups, 0);
    auto *dsumap = sumap.data().get();
    const auto *dhits = hits.data().get();

    int device_id = -1;

#if defined(XGBOOST_USE_CUDA)
    dh::safe_cuda(cudaGetDevice(&device_id));
#elif defined(XGBOOST_USE_HIP)
    dh::safe_cuda(hipGetDevice(&device_id));
#endif

    // For each group item compute the aggregated precision
    dh::LaunchN(nitems, nullptr, [=] __device__(uint32_t idx) {
      if (DetermineNonTrivialLabelLambda(idx)) {
        const auto group_idx = dgroup_idx[idx];
        const auto group_begin = dgroups[group_idx];
        const auto ridx = idx - group_begin;
        if (ridx < ecfg.topn) {
          atomicAdd(&dsumap[group_idx],
                    static_cast<double>(dhits[idx]) / (ridx + 1));
        }
      }
    });

    // Aggregate the group's item precisions
    dh::LaunchN(ngroups, nullptr, [=] __device__(uint32_t gidx) {
      auto nhits = dgroups[gidx + 1] ? dhits[dgroups[gidx + 1] - 1] : 0;
      if (nhits != 0) {
        dsumap[gidx] /= nhits;
      } else {
        if (ecfg.minus) {
          dsumap[gidx] = 0;
        } else {
          dsumap[gidx] = 1;
        }
      }
    });

#if defined(XGBOOST_USE_CUDA)
    return thrust::reduce(thrust::cuda::par(alloc), sumap.begin(), sumap.end());
#elif defined(XGBOOST_USE_HIP)
    return thrust::reduce(thrust::hip::par(alloc), sumap.begin(), sumap.end());
#endif
  }
};

XGBOOST_REGISTER_GPU_METRIC(PrecisionGpu, "pre")
.describe("precision@k for rank computed on GPU.")
.set_body([](const char* param) { return new EvalRankGpu<EvalPrecisionGpu>("pre", param); });

XGBOOST_REGISTER_GPU_METRIC(NDCGGpu, "ndcg")
.describe("ndcg@k for rank computed on GPU.")
.set_body([](const char* param) { return new EvalRankGpu<EvalNDCGGpu>("ndcg", param); });

XGBOOST_REGISTER_GPU_METRIC(MAPGpu, "map")
.describe("map@k for rank computed on GPU.")
.set_body([](const char* param) { return new EvalRankGpu<EvalMAPGpu>("map", param); });
}  // namespace metric
}  // namespace xgboost
