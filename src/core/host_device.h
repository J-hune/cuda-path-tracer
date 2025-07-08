#pragma once

#ifdef __CUDACC__
#   define RT_HOST_DEVICE __host__ __device__
#else
#   define RT_HOST_DEVICE
#endif
