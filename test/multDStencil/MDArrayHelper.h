#ifndef MDARRAY_H
#define MDARRAY_H

#define _CCM_ __host__ __device__ // CUDA Callable Member



template <class T> class MDArrayHelper
{
public:
    _CCM_ MDArrayHelper(T *data, int dim, int *dimSize);
    _CCM_ ~MDArrayHelper();

    _CCM_ void set(T val, int *index);
    _CCM_ T get(int *index);

    _CCM_ void reposition(int *index);

private:
    T *data;
    T *dataOrigin;
    int dim;
    int *dimSize;
    int *dimCoef;

    _CCM_ int getLinIndex(int *index);
    _CCM_ void calcDimCoef(const int *dimSize, int *dimCoef, int dim);

};

#include "MDArrayHelper.cut"


#endif