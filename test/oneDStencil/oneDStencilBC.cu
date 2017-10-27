#include <iostream>
#include <cstdlib>



/*
GPU kernel to perform 1 dim stencil 
on a data including boundary data
using shared memory

in: device array for input data including boundary
out: device array for output data including boundary unchanged
arraySize: size of in and out
wArr: weight array
wArrSize: size of wArr
*/
__global__
void stencilKernelShared (float *in, float *out, int arrSize, float *wArr, int wArrSize)
{
	// create index related local variables
    int midIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int radius = wArrSize / 2;
	
	// reposition input and output array pointers for simplicity
	in += radius;
	out += radius;

    // Arrange shared memory
    extern __shared__ float sharedMem[];

    float *sh_in = sharedMem;
	float *sh_wArr = &sh_in[blockDim.x + 2 * radius];
	
	// reposition sh_in array pointer
	sh_in += radius;

    // cache required part of the input array to shared memory
    sh_in[threadIdx.x] = in[midIndex]; // middle
    if (threadIdx.x < radius) sh_in[threadIdx.x - radius] = in[midIndex - radius]; // left
    if (threadIdx.x >= blockDim.x - radius) sh_in[threadIdx.x + radius] = in[midIndex + radius]; // right
    
    // copy boundaries unchanged to out
    if (midIndex < radius) out[midIndex - radius] = sh_in[threadIdx.x - radius]; // left
    if (midIndex >= blockDim.x - radius) out[midIndex + radius] = sh_in[threadIdx.x + radius]; // right

    // cache weight array to shared memory if nescessary
    float *wArrPtr;
    if (blockDim.x - 2 * radius >= wArrSize)
    {
        int startId = radius;
        if (threadIdx.x >= startID && threadIdx.x < startID + wArrSize) 
                sh_wArr[threadIdx.x - startID] = wArr[threadIdx.x - startID];
        wArrPtr = sh_wArr;
    }
    else
    {
        wArrPtr = wArr;
    }

    // reposition wArrPtr array pointer
    wArrPtr += radius;
    
    // synchronize threads before starting to access shared memory objects
    __syncthreads();
    
    // calculate output
    float result = 0;
    for (int i = -1 * radius; i <= radius; i++)
    {
        result += wArrPtr[i] * sh_in[threadIdx.x + i];
    }
    
    // write output
    out[midIndex] = result;
}

/*
in, out: input and output arrays including boundary of radius (wArrSize / 2) at both sides
arrSize: input and output array sizes
wArr: weight array
wArrSize: weight array size
*/
void applyStencil(float *in, float *out, int arrSize, float *wArr, int wArrSize)
{
	// declare and allocate device arrays
	float *d_in, *d_out, *d_wArr;

	int radius = wArrSize / 2;
	int dataSize = arrSize - 2 * radius; // without boundary

    cudaMalloc(&d_in, arrSize * sizeof(float));
    cudaMalloc(&d_out, arrSize * sizeof(float));
    cudaMalloc(&d_wArr, wArrSize * sizeof(float));

	// copy initial data from host to device
    cudaMemcpy(d_in, in, arrSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_wArr, wArr, wArrSize * sizeof(float), cudaMemcpyHostToDevice);

	// apply CUDA stencil kernel
    int nThread = 128;
	int sharedMemSize = (wArrSize + nThread + 2 * radius) * sizeof(float);
	
	stencilKernelShared <<< (dataSize + nThread - 1) / nThread, nThread, sharedMemSize >>> 
			(d_in, d_out, arrSize, d_wArr, wArrSize);

	// copy output data from device to host
	cudaMemcpy(out, d_out, arrSize * sizeof(float), cudaMemcpyDeviceToHost);
	
	// deallocate device arrays
	cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_wArr);
}


int main()
{
	// declare and allocate input, output, and weight arrays
    int dataSize = 1000000;
    int wArrSize = 15;
    
	int radius = wArrSize / 2;
	int arrSize = dataSize + 2 * radius;

    float *in = (float *) malloc(arrSize * sizeof(float));
    float *out = (float *) malloc(arrSize * sizeof(float));
    float *wArr = (float *) malloc(wArrSize * sizeof(float));

	// initialize input
	for (int i=0; i<dataSize; i++) in[i + radius] = (i % 2) + 1; // data
	for (int i=0; i<radius; i++) in[i] = in[i + dataSize + radius] = 0; // boundary

	// initialize output
	for (int i=0; i<arrSize; i++) out[i] = 0;
	
	// initialize weight array
    for (int i=0; i<wArrSize; i++) wArr[i] = (float) 1 / wArrSize;
    
	// apply stencil
	void applyStencil(in, out, arrSize, wArr, wArrSize);

	// display a portion of output
    for (int i=0; i<10; i++) std::cout << out[i] << " ";
    std::cout << std::endl;

    // deallocate arrays
    free(in);
    free(out);
    free(wArr);

    return 0;
}

