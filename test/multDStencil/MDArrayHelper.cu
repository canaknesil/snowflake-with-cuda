
#include "MDArrayHelper.h"


template <class T>
void MDArrayHelper<T>::calcDimCoef(const int *dimSize, int *dimCoef, int dim)
{
    for (int i=0; i<dim; i++)
    {
        int coef = 1;
        for (int j=i+1; j<dim; j++) coef *= dimSize[j];
        dimCoef[i] = coef;
    }
}

template <class T>
MDArrayHelper<T>::MDArrayHelper(T *data, int dim, int *dimSize)
{
    this->data = data;
    this->dataOrigin = data;
    this->dim = dim;

    this->dimSize = new int[dim];
    for (int i=0; i<dim; i++) this->dimSize[i] = dimSize[i];

    this->dimCoef = new int[dim];
    calcDimCoef(dimSize, this->dimCoef, dim);
}


template <class T>
MDArrayHelper<T>::~MDArrayHelper()
{
    delete[] dimSize;
    delete[] dimCoef;
}


template <class T>
int MDArrayHelper<T>::getLinIndex(int *index)
{
    int linIndex = 0;
    for (int i=0; i<dim; i++) linIndex += index[i] * dimCoef[i];
    return linIndex;
}


template <class T>
void MDArrayHelper<T>::set(T val, int *index)
{
    data[getLinIndex(index)] = val;
}


template <class T>
T MDArrayHelper<T>::get(int *index)
{
    return data[getLinIndex(index)];
}


template <class T>
void MDArrayHelper<T>::reposition(int *index)
{
    data = dataOrigin + getLinIndex(index);
}