
#include "MDStencil.h"




void performOp(int op, float *output, float *left, float *right, int size)
{
	for (int i=0; i<size; i++) output[i] = left[i] + right[i];
}