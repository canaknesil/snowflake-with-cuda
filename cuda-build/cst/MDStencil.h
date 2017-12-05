
#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3

void applyStencil(float *in, float *out, int *arrSize, float *wArr, int *wArrSize, int dim);
void performOp(int op, float *output, float *left, float *right, int size);