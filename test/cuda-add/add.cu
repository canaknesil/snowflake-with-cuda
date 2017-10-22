#include <iostream>
#include <math.h>
#include <cstdlib>

// function to add the elements of two arrays
void addCPU(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}


__global__
void addGPU_1_1(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];

}


__global__
void addGPU_N_1(int n, float *x, float *y)
{
  int nblock = gridDim.x;
  int blockId = blockIdx.x;

  int start = (n + nblock - 1) / nblock * blockId;
  int end =   (n + nblock - 1) / nblock * (blockId + 1);
  end = (end > n ? n : end);

  for (int i = start; i < end; i++)
  {
    y[i] = x[i] + y[i];
  }
}


__global__
void addGPU_1_N(int n, float *x, float *y)
{
  int nthread = blockDim.x;
  int threadId = threadIdx.x;

  int start = (n + nthread - 1) / nthread * threadId;
  int end =   (n + nthread - 1) / nthread * (threadId + 1);
  end = (end > n ? n : end);

  for (int i = start; i < end; i++)
  {
    y[i] = x[i] + y[i];
  }
}


__device__
void addInterval(int start, int end, float *x, float *y)
{
  for (int i = start; i < end; i++)
  {
    y[i] = x[i] + y[i];
  }
}


__global__
void addGPU_N_N(int n, float *x, float *y)
{
  int nBlock = gridDim.x;
  int nThreadPerBlock = blockDim.x;
  int blockId = blockIdx.x;
  int threadId = threadIdx.x;

  int nThread = nBlock * nThreadPerBlock;
  int arraySizePerThread = n / nThread;
  int arraySizePerBlock = arraySizePerThread * nThreadPerBlock;

  int start = blockId * arraySizePerBlock + threadId * arraySizePerThread;
  int end = start + arraySizePerThread;
  end = (end > n ? n : end);

  addInterval(start, end, x, y);
}


int mainCPU(void)
{
  int N = 1 << 20; // 1M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  addCPU(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete[] x;
  delete[] y;

  return 0;
}

int mainGPU()
{
    int N = 1000000; // 1M elements
  
    float *x, *dx;
    float *y, *dy;

    x = (float *) malloc(N * sizeof(float));
    y = (float *) malloc(N * sizeof(float));

    cudaMalloc(&dx, N*sizeof(float));
    cudaMalloc(&dy, N*sizeof(float));
  
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    cudaMemcpy(dx, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, N*sizeof(float), cudaMemcpyHostToDevice);
  
    // Run kernel on 1M elements on the CPU
    //addGPU_1_1 <<<4, 1>>> (N, dx, dy);
    //addGPU_N_1 <<<4, 1>>> (N, dx, dy);
    //addGPU_1_N <<<1, 4>>> (N, dx, dy);
    addGPU_N_N <<<2, 2>>> (N, dx, dy);

    cudaMemcpy(y, dy, N*sizeof(float), cudaMemcpyDeviceToHost);
  
    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
      maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;
  
    // Free memory
    cudaFree(dx);
    cudaFree(dy);

    free(x);
    free(y);
  
    return 0;
  
}

int main(void)
{
  mainGPU();
}