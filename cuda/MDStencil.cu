#include <iostream>
#include <cstdlib>
#include <cmath>
#include "MDArrayHelper.h"
#include "MDUtils.h"

using namespace std;

/*
Note: The kernel may need shared memory optimization
*/

/*
GPU kernel to perform dim dimentional stencil 
on a data including boundary data

in: input array for input data including boundary
out: output array for output data including boundary unchanged
arraySize: size of in and out for each dimention
wArr: weight array
wArrSize: size of wArr for each dimention

NOTE: Size of the data part of in and out (without boundaries) is a multiple of block side (number of thread per block)^(1/dim)
*/
__global__
void stencilKernelShared (float *in, float *out, int *arrSize, float *wArr, int *wArrSize, int dim)
{
    /*
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
    //if (midIndex < radius) out[midIndex - radius] = sh_in[threadIdx.x - radius]; // left
    //if (midIndex >= blockDim.x - radius) out[midIndex + radius] = sh_in[threadIdx.x + radius]; // right

    // cache weight array to shared memory if nescessary
    float *wArrPtr;
    if (blockDim.x - 2 * radius >= wArrSize)
    {
        int startId = radius;
        if (threadIdx.x >= startId && threadIdx.x < startId + wArrSize) 
                sh_wArr[threadIdx.x - startId] = wArr[threadIdx.x - startId];
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
    */
}

/*
in, out: input and output arrays including boundary of radius (wArrSize / 2) at both sides
arrSize: input and output array sizes for each dimention
wArr: weight array
wArrSize: weight array size for each dimention
*/
void applyStencil(float *in, float *out, int *arrSize, float *wArr, int *wArrSize, int dim)
{
    // calculate number of thread per block
    int maxNThread = 256;
    int blockSide = floor(pow(maxNThread, (float) 1 / dim));
    int nThread = pow(blockSide, dim);
    
    // create size related variables
    int *radius = (int *) alloca(dim);
    for(int i=0; i<dim; i++) radius[i] = wArrSize[i] / 2;

    int *dataSize = (int *) alloca(dim); // without boundary
    for(int i=0; i<dim; i++) dataSize[i] = arrSize[i] - 2 * radius[i];

    int *extArrSize = (int *) alloca(dim);
    for (int i=0; i<dim; i++)
    {
        int rest = dataSize[i] % blockSide;
        extArrSize[i] = (rest == 0 ? arrSize[i] : arrSize[i] + blockSide - rest);
    }

    int arrLinSize = 1;
    int extArrLinSize = 1;
    int wArrLinSize = 1;
    for (int i=0; i<dim; i++) 
    {
        arrLinSize *= arrSize[i];
        extArrLinSize *= extArrSize[i];
        wArrLinSize *= wArrSize[i];
    }

    // declare and allocate device arrays
	float *d_in, *d_out, *d_wArr;
    
    cudaMalloc(&d_in, extArrLinSize * sizeof(float));
    cudaMalloc(&d_out, extArrLinSize * sizeof(float));
    cudaMalloc(&d_wArr, wArrLinSize * sizeof(float));

    // copy initial data from host to device
    cudaMemcpy(d_in, in, arrLinSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_wArr, wArr, wArrLinSize * sizeof(float), cudaMemcpyHostToDevice);

    // apply CUDA stencil kernel	
    int *blockNPerDim = (int *) alloca(dim);
    for (int i=0; i<dim; i++) blockNPerDim[i] = (extArrSize[i] - 2 * radius[i]) / blockSide;

    int nBlock = 1;
    for (int i=0; i<dim; i++) nBlock *= blockNPerDim[i];

	stencilKernelShared <<< nBlock, nThread >>> 
			(d_in, d_out, extArrSize, d_wArr, wArrSize, dim);

	// copy output data from device to host
    cudaMemcpy(out, d_out, arrLinSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // copy boundaries unchanged to out
    // Note: This part needs optimization, this version iterates the entire data.
    MDArrayHelper<float> outHelper(out, dim, arrSize);
    MDArrayHelper<float> inHelper(in, dim, arrSize);

    int *i = (int *) alloca(dim);
    int *start = (int *) alloca(dim);
    int *end = (int *) alloca(dim);

    for (int i=0; i<dim; i++) 
    {
        start[i] = 0;
        end[i] = arrSize[i];
    }

    MDForHost(dim, i, start, end, [&] ()
    {
        bool pred = false; // boundary: true
        for (int a=0; a<dim; a++) if (i[a] < radius[a] || i[a] >= radius[a] + dataSize[a]) pred = true;

        if (pred)
        {
            outHelper.set(inHelper.get(i), i);
        }
    });
	
	// deallocate device arrays
	cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_wArr);
}



void print2D(float *arr, int *size)
{
    for (int i=0; i<size[0]; i++) 
    {
        for (int j=0; j<size[1]; j++) cout << arr[i * size[0] + j] << " ";
        cout << endl;
    }
}

void test2D()
{
    // declare and allocate input, output, and weight arrays
    int dim = 2;
    int dataSize[] = {5, 5};
    int wArrSize[] = {3, 3};
    
    int *radius = (int *) alloca(dim);
    for(int i=0; i<dim; i++) radius[i] = wArrSize[i] / 2;

    int *arrSize = (int *) alloca(dim);
    for (int i=0; i<dim; i++) arrSize[i] = dataSize[i] + 2 * radius[i];


    int arrLinSize = 1;
    int wArrLinSize = 1;
    for (int i=0; i<dim; i++) 
    {
        arrLinSize *= arrSize[i];
        wArrLinSize *= wArrSize[i];
    }

    float *in = new float[arrLinSize];
    float *out = new float[arrLinSize];
    float *wArr = new float[wArrLinSize];

    // initialize helpers
    MDArrayHelper<float> inHelper(in, dim, arrSize);
    MDArrayHelper<float> outHelper(out, dim, arrSize);
    MDArrayHelper<float> wHelper(wArr, dim, wArrSize);

    // reposision in and out helpers
    inHelper.reposition(radius);
    outHelper.reposition(radius);

    // initialize input array
    int *index = new int[dim];

    for (int linI = 0; linI<arrLinSize; linI++)
    {
        int *index = (int *) alloca(dim);
        inHelper.getCoords(index, linI);

        bool pred = true; // data: true, boundary: false
        for (int i=0; i<dim; i++) if (index[i] < 0 || index[i] >= dataSize[i]) pred = false;

        if (pred)
        {   // data
            int totIndex = 0;
            for (int i=0; i<dim; i++) totIndex += index[i];
            inHelper.set((totIndex % 2) + 1, index);
        }
        else
        {   //boundary
            inHelper.set(5, index);
        }
    }

    cout << "Input: " << endl;
    print2D(in, arrSize);
    cout << endl;

	// initialize output
    for (int linI=0; linI<arrLinSize; linI++) 
    {
        int *index = (int *) alloca(dim);
        outHelper.getCoords(index, linI);

        outHelper.set(0, index);
    }
	
	// initialize weight array
    for (int linI=0; linI<wArrLinSize; linI++) 
    {
        int *index = (int *) alloca(dim);
        wHelper.getCoords(index, linI);

        wHelper.set((float) 1 / wArrLinSize, index);
    }

    cout << "Weight Array: " << endl;
    print2D(wArr, wArrSize);
    cout << endl;
    
	// apply stencil
	applyStencil(in, out, arrSize, wArr, wArrSize, dim);

	cout << "Output: " << endl;
    print2D(out, arrSize);
    cout << endl;

    // deallocate arrays
    delete[] in;
    delete[] out;
    delete[] wArr;
}


int main()
{
    test2D();
    return 0;
}