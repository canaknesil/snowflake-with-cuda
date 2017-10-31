#ifndef MDARRAY_H
#define MDARRAY_H

#define _CCM_ __host__ __device__ // CUDA Callable Member


/*
Takes a pointer to a pre allocated contiguous array along with its dimention information.
And facilitates to make references to the array.

Must be used with nvcc compiler.
Can be used in both host and device code.
*/

template <class T> class MDArrayHelper
{
public:
    _CCM_ MDArrayHelper(T *data, int dim, int *dimSize);
    _CCM_ ~MDArrayHelper();

    _CCM_ void set(T val, int *index);
    _CCM_ T get(int *index);

    _CCM_ void reposition(int *index);

    _CCM_ int getLinIndex(int *index);
    _CCM_ void getCoords(int *index, int linIndex);

private:
    T *data;
    T *dataOrigin;
    int dim;
    int *dimSize;
    int *dimCoef;

    _CCM_ void calcDimCoef(const int *dimSize, int *dimCoef, int dim);

};

#include "MDArrayHelper.cut" // implementation


#endif