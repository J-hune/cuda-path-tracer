#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"

#include <cuda_runtime.h>

#include <vector>

// Kernel qui écrit l'index du thread dans un tableau
__global__ void write_thread_index(unsigned int* idx) {
   idx[threadIdx.x] = threadIdx.x;
}

unsigned int hello() {
   constexpr int kThreadCount = 32;

   std::vector<unsigned int> h_idx(kThreadCount);
   DeviceBuffer<unsigned int> d_idx(kThreadCount);

   // On lance le kernel avec un bloc de threads
   write_thread_index<<<1, kThreadCount>>>(d_idx.data());
   RT_CUDA_CHECK_LAUNCH();

   d_idx.download(h_idx);
   return h_idx.back();
}
