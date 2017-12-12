
#include "nodes.h"
#include <iostream>
#include "cst/MDStencil.h"

using namespace std;

#define NULL 0


StencilN::StencilN(string mesh, Node *body)
{
	this->mesh = mesh;
	this->body = body;
	evaluate(NULL, NULL);
}

StencilN::~StencilN() {}

void StencilN::evaluate(float **dummyOutput, int *dummySize)
{
	float *output;
	int size;

	body->evaluate(&output, &size);

	writeOut(output, size);
}

void StencilN::writeOut(float *arr, int size)
{	
	FILE *f = fopen(mesh.c_str(), "wb");
	fwrite(arr, sizeof(float), size, f);
	fclose(f);
}


StencilComponentN::StencilComponentN(string mesh, float* weights, int dim, int size, int *wSizes)
{
	this->mesh = mesh;
	this->weights = weights;
	this->dim = dim;
	this->size = size;
	this->wSizes = wSizes;
}

StencilComponentN::~StencilComponentN() {}

void StencilComponentN::evaluate(float **output, int *outputSize)
{
	float *input;
	int inputSize;
	int *dims;

	readIn(&input, &inputSize, &dims);

	*outputSize = inputSize;
	*output = new float[inputSize];
	for (int i=0; i<inputSize; i++) (*output)[i] = 0;
	
	//apply stencil
	applyStencil(input, *output, dims, weights, wSizes, dim);
}

void StencilComponentN::readIn(float **arr, int *size, int **dims)
{
	FILE *f = fopen(mesh.c_str(), "rb");
	
	float fSize;
	fread(&fSize, sizeof(float), 1, f);
	*size = (int) fSize;
	
	float *fdims = new float[dim];
	*dims = new int[dim];
	fread(fdims, sizeof(float), dim, f);
	for (int i=0; i<dim; i++) (*dims)[i] = (int) fdims[i];
	
	*arr = new float[*size];
	fread(*arr, sizeof(float), *size, f);
	
	fclose(f);
}


StencilOpN::StencilOpN(int op, Node *left, Node *right)
{
	this->op = op;
	this->left = left;
	this->right = right;
}

StencilOpN::~StencilOpN() {}

void StencilOpN::evaluate(float **output, int *size)
{
	float *leftOut, *rightOut;
	
	left->evaluate(&leftOut, size);
	right->evaluate(&rightOut, size);

	*output = new float[*size];
	performOp(op, *output, leftOut, rightOut, *size);
}



StencilN *Stencil(string mesh, Node *body)
{
	return new StencilN(mesh, body);
}

StencilComponentN *StencilComponent(string mesh, float *weights, int dim, int size, int *wSizes)
{
	return new StencilComponentN(mesh, weights, dim, size, wSizes);
}

StencilOpN *StencilOp(int op, Node *left, Node *right)
{
	return new StencilOpN(op, left, right);
}










