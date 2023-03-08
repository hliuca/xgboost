/*!
 * Copyright 2018 XGBoost contributors
 */
// Dummy file to keep the CUDA conditional compile trick.

#if !defined(XGBOOST_USE_CUDA)
#include "elementwise_metric.cu"
#elif !defined(XGBOOST_USE_HIP)
#include "elementwise_metric.hip"
#endif  // !defined(XGBOOST_USE_CUDA)
