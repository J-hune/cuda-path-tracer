#pragma once

#include <cstdlib>
#include <iostream>
#include <optix_stubs.h>
#include <optix_types.h>

[[noreturn]]
inline void report_optix_error(OptixResult result, const char* call, const char* file, int line) {
   std::cerr << "OptiX error at " << file << ":" << line << "\n  call: " << call
             << "\n  error: " << optixGetErrorName(result)
             << "\n  description: " << optixGetErrorString(result) << '\n';
   std::exit(EXIT_FAILURE);
}

#define RT_OPTIX_CHECK(call)                                              \
   do {                                                                   \
      const OptixResult rt_optix_result_ = (call);                        \
      if (rt_optix_result_ != OPTIX_SUCCESS) {                            \
         report_optix_error(rt_optix_result_, #call, __FILE__, __LINE__); \
      }                                                                   \
   } while (0)
