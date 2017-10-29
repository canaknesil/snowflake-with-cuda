#include <iostream>
#include "MDArrayHelper.h"


void printArr(int *arr, int size)
{
    for (int i=0; i<size; i++) std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main()
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
        int index[] = {0, i};
        std::cout << arr.get(index) << " ";
    }
    std::cout << std::endl;

    return 0;
}