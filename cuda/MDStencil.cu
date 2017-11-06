#include <iostream>
#include <cstdlib>
#include <cmath>
#include "MDArrayHelper.h"
#include "MDUtils.h"

using namespace std;



__device__ void MDForDevice(int dim, int *i, int *start, int *end, void (*body)())
{
    if (dim == 0) 
    {
        body();
        return;
    }

    for (i[0] = start[0]; i[0] < end[0]; i[0]++)
    {
        MDForDevice(dim - 1, i + 1, start + 1, end + 1, body);
    }
}


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
void stencilKernel (float *in, float *out, int *arrSize, float *wArr, int *wArrSize, int dim, int blockSide)
{
    // create index related local variables
    int *blockSize = new int[dim];
    for (int i=0; i<dim; i++) blockSize[i] = blockSide;

    int *radius = new int[dim];
    for (int i=0; i<dim; i++) radius[i] = wArrSize[i] / 2;

    int *dataSize = new int[dim];
    for (int i=0; i<dim; i++) dataSize[i] = arrSize[i] - 2 * radius[i];

    int *gridSize = new int[dim];
    for (int i=0; i<dim; i++) gridSize[i] = dataSize[i] / blockSide;
    
    MDArrayHelper<char> threadH(0, dim, blockSize);
    MDArrayHelper<char> blockH(0, dim, gridSize);
    
    int *threadIndex = new int[dim];
    threadH.getCoords(threadIndex, threadIdx.x);

    int *blockIndex = new int[dim];
    blockH.getCoords(blockIndex, blockIdx.x);

    // initilize helpers for data
    MDArrayHelper<float> inH(in, dim, arrSize);
    MDArrayHelper<float> outH(out, dim, arrSize);
    MDArrayHelper<float> wArrH(wArr, dim, wArrSize);

    // reposition helpers
    int *newPosition = new int[dim];
    for (int i=0; i<dim; i++) newPosition[i] = blockIndex[i] * blockSide + radius[i];

    inH.reposition(newPosition);
    outH.reposition(newPosition);
    wArrH.reposition(radius);

    delete[] newPosition;
    
    // calculate output
    int *index = new int[dim];
    int *start = new int[dim];
    int *end = new int[dim];

    for (int i=0; i<dim; i++)
    {
        start[i] = 0;
        end[i] = blockSide;
    }

    float result = 0;

    MDForDevice(dim, index, start, end, [] () 
    {

    });
    
    outH.set(blockSize[0], threadIndex);

    

    delete[] index;
    
    /*
	
	
    // calculate output
    float result = 0;
    for (int i = -1 * radius; i <= radius; i++)
    {
        result += wArrPtr[i] * sh_in[threadIdx.x + i];
    }
    
    // write output
    out[midIndex] = result;
    */
  
    // deallocations
    delete[] blockSize;
    delete[] radius;
    delete[] dataSize;
    delete[] gridSize;
    delete[] threadIndex;
    delete[] blockIndex;
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

    // create extended array
    float *extIn = new float[extArrLinSize];

    MDArrayHelper<float> extInH(extIn, dim, extArrSize);
    MDArrayHelper<float> inH(in, dim, arrSize);

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
        extInH.set(inH.get(i), i);
    });

    for (int i=0; i<dim; i++) 
    {
        start[i] = arrSize[i];
        end[i] = extArrSize[i];
    }

    MDForHost(dim, i, start, end, [&] ()
    {
        extInH.set(inH.get(i), i);
    });


    // declare and allocate device arrays
    float *d_in, *d_out, *d_wArr;
    int *d_arrSize, *d_wArrSize;
    
    cudaMalloc(&d_in, extArrLinSize * sizeof(float));
    cudaMalloc(&d_out, extArrLinSize * sizeof(float));
    cudaMalloc(&d_wArr, wArrLinSize * sizeof(float));
    cudaMalloc(&d_arrSize, dim * sizeof(int));
    cudaMalloc(&d_wArrSize, dim * sizeof(int));

    // copy initial data from host to device
    cudaMemcpy(d_in, extIn, extArrLinSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wArr, wArr, wArrLinSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrSize, extArrSize, dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wArrSize, wArrSize, dim * sizeof(int), cudaMemcpyHostToDevice);

    // apply CUDA stencil kernel	
    int *blockNPerDim = (int *) alloca(dim);
    for (int i=0; i<dim; i++) blockNPerDim[i] = (extArrSize[i] - 2 * radius[i]) / blockSide;

    int nBlock = 1;
    for (int i=0; i<dim; i++) nBlock *= blockNPerDim[i];

	stencilKernel <<< nBlock, nThread >>> 
			(d_in, d_out, d_arrSize, d_wArr, d_wArrSize, dim, blockSide);

    // copy output data from device to host
    float *extOut = new float[extArrLinSize];
    cudaMemcpy(extOut, d_out, extArrLinSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // create out out of extOut and copy boundaries unchanged
    // Note: This part needs optimization, this version iterates the entire data.
    MDArrayHelper<float> outH(out, dim, arrSize);
    MDArrayHelper<float> extOutH(extOut, dim, extArrSize);

    for (int i=0; i<dim; i++) 
    {
        start[i] = 0;
        end[i] = arrSize[i];
    }

    MDForHost(dim, i, start, end, [&] ()
    {
        bool pred = false; // boundary: true
        for (int a=0; a<dim; a++) if (i[a] < radius[a] || i[a] >= radius[a] + dataSize[a]) pred = true;

        if (pred) outH.set(inH.get(i), i);
        else outH.set(extOutH.get(i), i);
    });
	
	// deallocate device arrays
	cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_wArr);
    cudaFree(d_arrSize);
    cudaFree(d_wArrSize);

    delete[] extIn;
    delete[] extOut;
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