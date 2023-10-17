/**
 * Copyright 2017-2023 XGBoost contributors
 */
#pragma once

#if defined(XGBOOST_USE_HIP)

#define cudaSuccess                  hipSuccess
#define cudaGetLastError             hipGetLastError

#define cudaStream_t                 hipStream_t
#define cudaStreamCreate             hipStreamCreate
#define cudaStreamCreateWithFlags    hipStreamCreateWithFlags
#define cudaStreamDestroy            hipStreamDestroy
#define cudaStreamWaitEvent          hipStreamWaitEvent
#define cudaStreamSynchronize        hipStreamSynchronize
#define cudaStreamPerThread          hipStreamPerThread
#define cudaStreamLegacy             hipStreamLegacy

#define cudaEvent_t                  hipEvent_t
#define cudaEventCreate              hipEventCreate
#define cudaEventCreateWithFlags     hipEventCreateWithFlags
#define cudaEventDestroy             hipEventDestroy

#define cudaGetDevice                hipGetDevice
#define cudaSetDevice                hipSetDevice
#define cudaGetDeviceCount           hipGetDeviceCount
#define cudaDeviceSynchronize        hipDeviceSynchronize

#define cudaGetDeviceProperties      hipGetDeviceProperties
#define cudaDeviceGetAttribute       hipDeviceGetAttribute

#define cudaMallocHost               hipMallocHost
#define cudaFreeHost                 hipFreeHost
#define cudaMalloc                   hipMalloc
#define cudaFree                     hipFree

#define cudaMemcpy                   hipMemcpy
#define cudaMemcpyAsync              hipMemcpyAsync
#define cudaMemcpyDefault            hipMemcpyDefault
#define cudaMemcpyHostToDevice       hipMemcpyHostToDevice
#define cudaMemcpyHostToHost         hipMemcpyHostToHost
#define cudaMemcpyDeviceToHost       hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice     hipMemcpyDeviceToDevice
#define cudaMemsetAsync              hipMemsetAsync
#define cudaMemset                   hipMemset

#define cudaPointerAttributes        hipPointerAttribute_t 
#define cudaPointerGetAttributes     hipPointerGetAttributes

#define cudaMemGetInfo               hipMemGetInfo
#define cudaFuncSetAttribute         hipFuncSetAttribute

#define cudaDevAttrMultiProcessorCount                hipDeviceAttributeMultiprocessorCount
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor

#endif
