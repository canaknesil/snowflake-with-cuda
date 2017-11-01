#ifndef MDUTILS_H
#define MDUTILS_H

#define _CCM_ __host__ __device__ // CUDA Callable Member

#include <functional>


_CCM_ void MDFor(int dim, int *i, int *start, int *end, std::function<void ()> body);


#endif