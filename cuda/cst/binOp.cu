
#include "MDStencil.h"



__global__
void binOpKernel(int op, float *output, float *left, float *right, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	switch(op)
	{
		case ADD:
			output[i] = left[i] + right[i];
			break;
			
		case SUB:
			output[i] = left[i] - right[i];
			break;
			
		case MUL:
			output[i] = left[i] * right[i];
			break;
			
		case DIV:
			output[i] = left[i] / right[i];
			break;
	}
}


#define THREAD_N 128

void performOp(int op, float *output, float *left, float *right, int size)
{
	int blockN = (size + THREAD_N - 1) / size;
	int extSize = blockN * THREAD_N;
	
	float *d_out, *d_left, *d_right;
	
	cudaMalloc(&d_out, extSize * sizeof(float));
	cudaMalloc(&d_left, extSize * sizeof(float));
	cudaMalloc(&d_right, extSize * sizeof(float));
	
	cudaMemcpy(d_left, left, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right, right, size * sizeof(float), cudaMemcpyHostToDevice);
	
	binOpKernel <<< blockN, THREAD_N >>> (op, d_out, d_left, d_right, extSize);
	
	cudaMemcpy(output, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_out);
	cudaFree(d_left);
	cudaFree(d_right);
}