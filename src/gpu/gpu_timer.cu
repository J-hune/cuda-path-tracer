#include "gpu/cuda_check.h"
#include "gpu/gpu_timer.h"

#include <cuda_runtime.h>

GpuTimer::GpuTimer() {
   RT_CUDA_CHECK(cudaEventCreate(&start_));
   RT_CUDA_CHECK(cudaEventCreate(&stop_));
}

GpuTimer::~GpuTimer() {
   cudaEventDestroy(start_);
   cudaEventDestroy(stop_);
}

void GpuTimer::start() {
   RT_CUDA_CHECK(cudaEventRecord(start_));
}

void GpuTimer::stop() {
   RT_CUDA_CHECK(cudaEventRecord(stop_));
}

float GpuTimer::elapsed_milliseconds() const {
   RT_CUDA_CHECK(cudaEventSynchronize(stop_));
   float milliseconds = 0.0f;
   RT_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
   return milliseconds;
}
