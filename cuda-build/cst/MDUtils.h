#ifndef MDUTILS_H
#define MDUTILS_H

#define _CCM_ __host__ __device__ // CUDA Callable Member

#include <functional>


__host__ void MDForHost(int dim, int *i, int *start, int *end, std::function<void ()> body);
//__device__ void MDForDevice(int dim, int *i, int *start, int *end, void (*body)());



#endif