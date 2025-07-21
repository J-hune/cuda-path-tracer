#pragma once

#include <cuda_runtime.h>

// Chronomètre GPU à base d'événements CUDA : mesure le temps passé par le GPU entre start() et
// stop(), sans compter l'attente du CPU ni la synchronisation de l'affichage
class GpuTimer {
public:
   GpuTimer();
   ~GpuTimer();
   GpuTimer(const GpuTimer&) = delete;
   GpuTimer& operator=(const GpuTimer&) = delete;
   GpuTimer(GpuTimer&&) = delete;
   GpuTimer& operator=(GpuTimer&&) = delete;

   void start();
   void stop();
   // Attend la fin du travail mesuré, puis renvoie sa durée en millisecondes
   [[nodiscard]] float elapsed_milliseconds() const;

private:
   cudaEvent_t start_ = nullptr;
   cudaEvent_t stop_ = nullptr;
};
