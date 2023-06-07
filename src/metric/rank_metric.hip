/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include <dmlc/registry.h>
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/reduce.h>                      // for reduce

#include <algorithm>                            // for transform
#include <cstddef>                              // for size_t
#include <memory>                               // for shared_ptr
#include <vector>                               // for vector

#include "../common/cuda_context.hip.h"           // for CUDAContext
#include "../common/device_helpers.hip.h"         // for MakeTransformIterator
#include "../common/optional_weight.h"          // for MakeOptionalWeights
#include "../common/ranking_utils.hip.h"          // for CalcQueriesDCG, NDCGCache
#include "metric_common.h"
#include "rank_metric.h"
#include "xgboost/base.h"                // for XGBOOST_DEVICE
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for MakeTensorView
#include "xgboost/logging.h"             // for CHECK
#include "xgboost/metric.h"

#if defined(XGBOOST_USE_HIP)
#include <hipcub/hipcub.hpp>

namespace cub = hipcub;
<<<<<<< HEAD
#endif

namespace xgboost::metric {
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

XGBOOST_REGISTER_GPU_METRIC(PrecisionGpu, "pre")
.describe("precision@k for rank computed on GPU.")
.set_body([](const char* param) { return new EvalRankGpu<EvalPrecisionGpu>("pre", param); });

namespace cuda_impl {
=======
#endif

namespace xgboost::metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(rank_metric_gpu);

namespace cuda_impl {
PackedReduceResult PreScore(Context const *ctx, MetaInfo const &info,
                            HostDeviceVector<float> const &predt,
                            std::shared_ptr<ltr::PreCache> p_cache) {
  auto d_gptr = p_cache->DataGroupPtr(ctx);
  auto d_label = info.labels.View(ctx->gpu_id).Slice(linalg::All(), 0);

  predt.SetDevice(ctx->gpu_id);
  auto d_rank_idx = p_cache->SortedIdx(ctx, predt.ConstDeviceSpan());
  auto topk = p_cache->Param().TopK();
  auto d_weight = common::MakeOptionalWeights(ctx, info.weights_);

  auto it = dh::MakeTransformIterator<double>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) {
        auto g = dh::SegmentId(d_gptr, i);
        auto g_begin = d_gptr[g];
        auto g_end = d_gptr[g + 1];
        i -= g_begin;
        auto g_label = d_label.Slice(linalg::Range(g_begin, g_end));
        auto g_rank = d_rank_idx.subspan(g_begin, g_end - g_begin);
        double y = g_label(g_rank[i]);
        auto n = std::min(static_cast<std::size_t>(topk), g_label.Size());
        double w{d_weight[g]};
        if (i >= n) {
          return 0.0;
        }
        return y / static_cast<double>(n) * w;
      });

  auto cuctx = ctx->CUDACtx();
  auto pre = p_cache->Pre(ctx);
  thrust::fill_n(cuctx->CTP(), pre.data(), pre.size(), 0.0);

  std::size_t bytes;
  cub::DeviceSegmentedReduce::Sum(nullptr, bytes, it, pre.data(), p_cache->Groups(), d_gptr.data(),
                                  d_gptr.data() + 1, cuctx->Stream());
  dh::TemporaryArray<char> temp(bytes);
  cub::DeviceSegmentedReduce::Sum(temp.data().get(), bytes, it, pre.data(), p_cache->Groups(),
                                  d_gptr.data(), d_gptr.data() + 1, cuctx->Stream());

  auto w_it =
      dh::MakeTransformIterator<double>(thrust::make_counting_iterator(0ul),
                                        [=] XGBOOST_DEVICE(std::size_t g) { return d_weight[g]; });
  auto n_weights = p_cache->Groups();
  auto sw = dh::Reduce(cuctx->CTP(), w_it, w_it + n_weights, 0.0, thrust::plus<double>{});
  auto sum =
      dh::Reduce(cuctx->CTP(), dh::tcbegin(pre), dh::tcend(pre), 0.0, thrust::plus<double>{});
  auto result = PackedReduceResult{sum, sw};
  return result;
}

>>>>>>> sync-sep-2023Jun01
PackedReduceResult NDCGScore(Context const *ctx, MetaInfo const &info,
                             HostDeviceVector<float> const &predt, bool minus,
                             std::shared_ptr<ltr::NDCGCache> p_cache) {
  CHECK(p_cache);

  auto const &p = p_cache->Param();
  auto d_weight = common::MakeOptionalWeights(ctx, info.weights_);
  if (!d_weight.Empty()) {
    CHECK_EQ(d_weight.weights.size(), p_cache->Groups());
  }
  auto d_label = info.labels.View(ctx->gpu_id).Slice(linalg::All(), 0);
  predt.SetDevice(ctx->gpu_id);
  auto d_predt = linalg::MakeTensorView(ctx, predt.ConstDeviceSpan(), predt.Size());

  auto d_group_ptr = p_cache->DataGroupPtr(ctx);

  auto d_inv_idcg = p_cache->InvIDCG(ctx);
  auto d_sorted_idx = p_cache->SortedIdx(ctx, d_predt.Values());
  auto d_out_dcg = p_cache->Dcg(ctx);

  ltr::cuda_impl::CalcQueriesDCG(ctx, d_label, d_sorted_idx, p.ndcg_exp_gain, d_group_ptr, p.TopK(),
                                 d_out_dcg);
<<<<<<< HEAD
=======

>>>>>>> sync-sep-2023Jun01
  auto it = dh::MakeTransformIterator<PackedReduceResult>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) {
        if (d_inv_idcg(i) <= 0.0) {
          return PackedReduceResult{minus ? 0.0 : 1.0, static_cast<double>(d_weight[i])};
        }
        return PackedReduceResult{d_out_dcg(i) * d_inv_idcg(i) * d_weight[i],
                                  static_cast<double>(d_weight[i])};
      });
  auto pair = thrust::reduce(ctx->CUDACtx()->CTP(), it, it + d_out_dcg.Size(),
                             PackedReduceResult{0.0, 0.0});
  return pair;
}

PackedReduceResult MAPScore(Context const *ctx, MetaInfo const &info,
                            HostDeviceVector<float> const &predt, bool minus,
                            std::shared_ptr<ltr::MAPCache> p_cache) {
  auto d_group_ptr = p_cache->DataGroupPtr(ctx);
  auto d_label = info.labels.View(ctx->gpu_id).Slice(linalg::All(), 0);

  predt.SetDevice(ctx->gpu_id);
  auto d_rank_idx = p_cache->SortedIdx(ctx, predt.ConstDeviceSpan());
  auto key_it = dh::MakeTransformIterator<std::size_t>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) { return dh::SegmentId(d_group_ptr, i); });

  auto get_label = [=] XGBOOST_DEVICE(std::size_t i) {
    auto g = key_it[i];
    auto g_begin = d_group_ptr[g];
    auto g_end = d_group_ptr[g + 1];
    i -= g_begin;
    auto g_label = d_label.Slice(linalg::Range(g_begin, g_end));
    auto g_rank = d_rank_idx.subspan(g_begin, g_end - g_begin);
    return g_label(g_rank[i]);
  };
  auto it = dh::MakeTransformIterator<double>(thrust::make_counting_iterator(0ul), get_label);

  auto cuctx = ctx->CUDACtx();
  auto n_rel = p_cache->NumRelevant(ctx);
  thrust::inclusive_scan_by_key(cuctx->CTP(), key_it, key_it + d_label.Size(), it, n_rel.data());

  double topk = p_cache->Param().TopK();
  auto map = p_cache->Map(ctx);
  thrust::fill_n(cuctx->CTP(), map.data(), map.size(), 0.0);
  {
    auto val_it = dh::MakeTransformIterator<double>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) {
          auto g = key_it[i];
          auto g_begin = d_group_ptr[g];
          auto g_end = d_group_ptr[g + 1];
          i -= g_begin;
          if (i >= topk) {
            return 0.0;
          }

          auto g_label = d_label.Slice(linalg::Range(g_begin, g_end));
          auto g_rank = d_rank_idx.subspan(g_begin, g_end - g_begin);
          auto label = g_label(g_rank[i]);

          auto g_n_rel = n_rel.subspan(g_begin, g_end - g_begin);
          auto nhits = g_n_rel[i];
          return nhits / static_cast<double>(i + 1) * label;
        });

    std::size_t bytes;
    cub::DeviceSegmentedReduce::Sum(nullptr, bytes, val_it, map.data(), p_cache->Groups(),
                                    d_group_ptr.data(), d_group_ptr.data() + 1, cuctx->Stream());
    dh::TemporaryArray<char> temp(bytes);
    cub::DeviceSegmentedReduce::Sum(temp.data().get(), bytes, val_it, map.data(), p_cache->Groups(),
                                    d_group_ptr.data(), d_group_ptr.data() + 1, cuctx->Stream());
  }

  PackedReduceResult result{0.0, 0.0};
  {
    auto d_weight = common::MakeOptionalWeights(ctx, info.weights_);
    if (!d_weight.Empty()) {
      CHECK_EQ(d_weight.weights.size(), p_cache->Groups());
    }
    auto val_it = dh::MakeTransformIterator<PackedReduceResult>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t g) {
          auto g_begin = d_group_ptr[g];
          auto g_end = d_group_ptr[g + 1];
          auto g_n_rel = n_rel.subspan(g_begin, g_end - g_begin);
          if (!g_n_rel.empty() && g_n_rel.back() > 0.0) {
            return PackedReduceResult{map[g] * d_weight[g] / std::min(g_n_rel.back(), topk),
                                      static_cast<double>(d_weight[g])};
          }
          return PackedReduceResult{minus ? 0.0 : 1.0, static_cast<double>(d_weight[g])};
        });
    result =
        thrust::reduce(cuctx->CTP(), val_it, val_it + map.size(), PackedReduceResult{0.0, 0.0});
  }
  return result;
}
}  // namespace cuda_impl
}  // namespace xgboost::metric
