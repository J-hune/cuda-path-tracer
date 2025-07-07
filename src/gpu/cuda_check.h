#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

[[noreturn]]
inline void report_cuda_error(cudaError_t error, const char* call, const char* file, int line) {
   std::cerr << "CUDA error at " << file << ":" << line << "\n  call: " << call
             << "\n  error: " << cudaGetErrorName(error)
             << "\n  description: " << cudaGetErrorString(error) << '\n';

   std::exit(EXIT_FAILURE);
}

#define RT_CUDA_CHECK(call)                                             \
   do {                                                                 \
      cudaError_t rt_cuda_status_ = (call);                             \
      if (rt_cuda_status_ != cudaSuccess) {                             \
         report_cuda_error(rt_cuda_status_, #call, __FILE__, __LINE__); \
      }                                                                 \
   } while (0)

#define RT_CUDA_CHECK_LAUNCH()                                                    \
   do {                                                                           \
      cudaError_t rt_cuda_status_ = cudaGetLastError();                           \
      if (rt_cuda_status_ != cudaSuccess) {                                       \
         report_cuda_error(rt_cuda_status_, "kernel launch", __FILE__, __LINE__); \
      }                                                                           \
   } while (0)
