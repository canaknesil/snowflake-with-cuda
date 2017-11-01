#include <iostream>
#include "MDUtils.h"

using namespace std;


#define DIM 3


__global__
void testKernel(int *out)
{
    int i[DIM] = {0};
    int start[] = {2, 2, 2};
    int end[] = {5, 5, 5};

    int count = 0;

    MDFor(DIM, i, start, end, [&] ()
    {
        for (int a=0; a<DIM; a++) out[count++] = i[a];
    });
}


void testDevice()
{
    int out[81];
    int *d_out;
    cudaMalloc(&d_out, 81 * sizeof(int));

    testKernel <<1, 1>> (d_out);

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

    MDFor(DIM, i, start, end, [&] ()
    {
        for (int a=0; a<DIM; a++) cout << i[a] << " ";
        cout << endl;
    });
}


int main()
{
    testHost();
    testDevice();
    
    return 0;
}