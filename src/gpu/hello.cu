#include "gpu/cuda_check.h"

#include <vector>

// Kernel qui écrit l'index du thread dans un tableau
__global__ void write_thread_index(unsigned int* idx) {
   idx[threadIdx.x] = threadIdx.x;
}

unsigned int hello() {
   constexpr int kThreadCount = 32;

   std::vector<unsigned int> h_idx(kThreadCount);
   unsigned int* d_idx = nullptr;

   // On alloue la mémoire sur le GPU pour stocker les indices des threads
   RT_CUDA_CHECK(cudaMalloc(&d_idx, kThreadCount * sizeof(unsigned int)));

   // On lance le kernel avec un bloc de threads
   write_thread_index<<<1, kThreadCount>>>(d_idx);
   RT_CUDA_CHECK_LAUNCH();

   RT_CUDA_CHECK(
      cudaMemcpy(h_idx.data(), d_idx, kThreadCount * sizeof(unsigned int), cudaMemcpyDeviceToHost)
   );

   RT_CUDA_CHECK(cudaFree(d_idx));

   return h_idx.back();
}
