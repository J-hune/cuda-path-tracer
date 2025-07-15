#pragma once
#include "gpu/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <span>
#include <utility>

template <typename T>
class DeviceBuffer {
public:
   explicit DeviceBuffer(std::size_t size):
      size_(size) {
      RT_CUDA_CHECK(cudaMalloc(&ptr_, size * sizeof(T)));
   }

   ~DeviceBuffer() { release(); }

   // Copie interdite (construction et affectation) : un seul propriétaire de la mémoire GPU
   DeviceBuffer(const DeviceBuffer&) = delete;
   DeviceBuffer& operator=(const DeviceBuffer&) = delete;

   DeviceBuffer(DeviceBuffer&& other) noexcept:
      ptr_(std::exchange(other.ptr_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

   DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
      if (this != &other) {
         release();

         ptr_ = std::exchange(other.ptr_, nullptr);
         size_ = std::exchange(other.size_, 0);
      }

      return *this;
   }

   [[nodiscard]] T* data() noexcept { return ptr_; }

   [[nodiscard]] const T* data() const noexcept { return ptr_; }

   [[nodiscard]] std::size_t size() const noexcept { return size_; }

   [[nodiscard]] std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }

   void upload(std::span<const T> host) {
      check_host_size(host.size());
      RT_CUDA_CHECK(cudaMemcpy(ptr_, host.data(), size_bytes(), cudaMemcpyHostToDevice));
   }

   void download(std::span<T> host) const {
      check_host_size(host.size());
      RT_CUDA_CHECK(cudaMemcpy(host.data(), ptr_, size_bytes(), cudaMemcpyDeviceToHost));
   }

private:
   T* ptr_ = nullptr;
   std::size_t size_ = 0;

   void release() noexcept {
      if (ptr_) {
         const cudaError_t status = cudaFree(ptr_);
         if (status != cudaSuccess && status != cudaErrorCudartUnloading) {
            std::cerr << "CUDA error during cudaFree: " << cudaGetErrorName(status) << " - "
                      << cudaGetErrorString(status) << '\n';
         }
      }

      ptr_ = nullptr;
      size_ = 0;
   }

   void check_host_size(std::size_t host_size) const {
      if (host_size != size_) {
         std::cerr << "Error: host data size (" << host_size
                   << ") does not match device buffer size (" << size_ << ")\n";
         std::abort();
      }
   }
};
