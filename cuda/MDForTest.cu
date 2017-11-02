#include <iostream>
#include "MDUtils.h"

using namespace std;


#define DIM 3


__device__
void MDForDevice(int dim, int *i, int *start, int *end, void (*body)(void **args), void **args)
{
    if (dim == 0) 
    {
        body(args);
        return;
    }

    for (i[0] = start[0]; i[0] < end[0]; i[0]++)
    {
        MDForDevice(dim - 1, i + 1, start + 1, end + 1, body, args);
    }
}


__global__
void testKernel(int *out)
{
    int i[DIM] = {0};
    int start[] = {2, 2, 2};
    int end[] = {5, 5, 5};

    int count = 0;

    void *args[3];
    args[0] = &i;
    args[1] = &out;
    args[2] = &count;

    auto body = [] (void **args)
    {
        int **i = (int **) args[0];
        int **out = (int **) args[1];
        int *count = (int *) args[2];

        for (int a=0; a<DIM; a++) (*out)[*count++] = (*i)[a];
    };

    MDForDevice(DIM, i, start, end, body, args);
}



void testDevice()
{
    int out[81];
    
    int *d_out;

    cudaMalloc(&d_out, 81 * sizeof(int));
    testKernel <<<1, 1>>> (d_out);
    cudaMemcpy(out, d_out, 81 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i<27; i++) 
    {
        for (int j=0; j<3; j++) cout << out[3 * i + j] << " ";
        cout << endl;
    } 
}


void testHost()
{
    int i[DIM] = {0};
    int start[] = {2, 2, 2};
    int end[] = {5, 5, 5};

    MDForHost(DIM, i, start, end, [&] ()
    {
        for (int a=0; a<DIM; a++) cout << i[a] << " ";
        cout << endl;
    });
}


int main()
{
    testHost();
    //testDevice();
    
    return 0;
}