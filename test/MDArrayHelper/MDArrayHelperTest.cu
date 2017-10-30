#include <iostream>
#include "MDArrayHelper.h"


void printArr(int *arr, int size)
{
    for (int i=0; i<size; i++) std::cout << arr[i] << " ";
    std::cout << std::endl;
}

void testOnHost()
{
    int orjArr[] = {1, 2, 3,
                    4, 5, 6,
                    7, 8, 9};

    int dim = 2;
    int dimSize[] = {3, 3};
    int linSize = 9;

    MDArrayHelper<int> arr(orjArr, dim, dimSize);

    for (int i=0; i<dimSize[1]; i++) 
    {
        int index[] = {1, i};
        arr.set(0, index);
    }

    printArr(orjArr, linSize);

    for (int i=0; i<dimSize[1]; i++) 
    {
        int index[] = {0, i};
        std::cout << arr.get(index) << " ";
    }
    std::cout << std::endl;

    int index[] = {1, 0};
    arr.reposition(index);

    for (int i=0; i<dimSize[1]; i++) 
    {
        int index[] = {1, i};
        std::cout << arr.get(index) << " ";
    }
    std::cout << std::endl;
}

__global__ void testKernel(int *out)
{
    int dim = 2;
    int dimSize[] = {3, 3};

    MDArrayHelper<int> arr(out, dim, dimSize);

    for (int i=0; i<dimSize[1]; i++) 
    {
        int index[] = {1, i};
        arr.set(0, index);
    }

    int index[] = {1, 0};
    arr.reposition(index);

    for (int i=0; i<dimSize[1]; i++) 
    {
        int index[] = {1, i};
        arr.set(1, index);
    }
}

void testOnDevice()
{
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    int *d_arr;
    cudaMalloc(&d_arr, 9 * sizeof(int));

    cudaMemcpy(d_arr, arr, 9 * sizeof(int), cudaMemcpyHostToDevice);

    testKernel <<<1, 1>>> (d_arr);

    cudaMemcpy(arr, d_arr, 9 * sizeof(int), cudaMemcpyDeviceToHost);

    printArr(arr, 9);
}

int main()
{
    testOnHost();
    testOnDevice();
    return 0;
}

/* Standart output:

1 2 3 0 0 0 7 8 9
1 2 3
7 8 9
1 2 3 0 0 0 1 1 1

*/