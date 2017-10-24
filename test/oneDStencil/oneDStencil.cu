#include <iostream>
#include <cstdlib>


// GPU kernel without shared memory usage
__global__
void stencilKernel (int arrSize, float *in, float *out, int wArrSize, float *wArr)
{
    int midIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int radius = wArrSize / 2;
    
    float result = 0;
    for (int i = -1 * radius; i <= radius; i++)
    {
        int arrIndex = midIndex + i;
        if (arrIndex >= 0 && arrIndex < arrSize) result += wArr[i + radius] * in[arrIndex];
    }

    if (midIndex >= 0 && midIndex < arrSize) out[midIndex] = result;
}


// GPU kernel with shared memory usage
__global__
void stencilKernelShared (int arrSize, float *in, float *out, int wArrSize, float *wArr)
{
    int midIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int radius = wArrSize / 2;
    
    // Arrange shared memory
    extern __shared__ float sharedMem[];

    float *sh_in = sharedMem;
    float *sh_wArr = &sh_in[blockDim.x + 2 * radius];


    sh_in[threadIdx.x + radius] = in[midIndex];
    if (threadIdx.x < radius) sh_in[threadIdx.x] = (midIndex - radius < 0 ? 0 : in[midIndex - radius]);
    if (threadIdx.x >= blockDim.x - radius) sh_in[threadIdx.x + 2 * radius] = (midIndex + radius >= arrSize ? 0 : in[midIndex + radius]);
    
    float *wArrPtr;
    if (blockDim.x - 2 * radius >= wArrSize)
    {
        if (threadIdx.x >= radius && threadIdx.x < radius + wArrSize) sh_wArr[threadIdx.x - radius] = wArr[threadIdx.x - radius];
        wArrPtr = sh_wArr;
    }
    else
    {
        wArrPtr = wArr;
    }
    
    __syncthreads();
    
    // calculate output
    float result = 0;
    for (int i = -1 * radius; i <= radius; i++)
    {
        result += wArrPtr[i + radius] * sh_in[threadIdx.x + i + radius];
    }
    
    // write output
    if (midIndex >= 0 && midIndex < arrSize) out[midIndex] = result;
}



int main()
{
    int arrSize = 1000000;
    int wArrSize = 15;

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

    //stencilKernel <<< (arrSize + nThread - 1) / nThread, nThread >>> (arrSize, d_in, d_out, wArrSize, d_wArr);

    int radius = wArrSize / 2;
    int sharedMemSize = (wArrSize + nThread + 2 * radius) * sizeof(float);
    stencilKernelShared <<< (arrSize + nThread - 1) / nThread, nThread, sharedMemSize >>> (arrSize, d_in, d_out, wArrSize, d_wArr);


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

