#include <iostream>
#include <cstdlib>



__global__
void stencilKernel (int arrSize, float *in, float *out, int wArrSize, float *wArr)
{
    int midIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int radius = wArrSize / 2;
    
    int result = 0;
    for (int i = -1 * radius; i <= radius; i++)
    {
        int arrIndex = midIndex + i;
        if (arrIndex >= 0 && arrIndex < arrSize) result += wArr[i + radius] * in[arrIndex];
    }

    if (midIndex >= 0 && midIndex < arrSize) out[midIndex] = result;
    
    //if (midIndex >= 0 && midIndex < arrSize) out[midIndex] = radius;
    //if (midIndex >= 0 && midIndex < arrSize) out[midIndex] = in[midIndex];
}



int main()
{
    int arrSize = 10000;
    int wArrSize = 5;

    float *in = (float *) malloc(arrSize * sizeof(float));
    float *out = (float *) malloc(arrSize * sizeof(float));
    float *wArr = (float *) malloc(wArrSize * sizeof(float));

    for (int i=0; i<arrSize; i++) in[i] = i % 2;
    for (int i=0; i<wArrSize; i++) wArr[i] = (float) 1 / wArrSize;
    

    float *d_in, *d_out, *d_wArr;
    cudaMalloc(&d_in, arrSize * sizeof(float));
    cudaMalloc(&d_out, arrSize * sizeof(float));
    cudaMalloc(&d_wArr, wArrSize * sizeof(float));

    cudaMemcpy(d_in, in, arrSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wArr, wArr, wArrSize * sizeof(float), cudaMemcpyHostToDevice);


    int nThread = 128;
    stencilKernel <<< (arrSize + nThread - 1) / nThread, nThread >>> (arrSize, d_in, d_out, wArrSize, d_wArr);


    cudaMemcpy(out, d_out, arrSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<10; i++)
    {
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_wArr);
    free(in);
    free(out);
    free(wArr);

    return 0;
}

