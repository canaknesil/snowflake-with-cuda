
#include "nodes.h"
#include <iostream>

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
	//TODO
	for (int i=0; i<size; i++) cout << arr[i] << " ";
	cout << endl;
}


StencilComponentN::StencilComponentN(string mesh, float* weights, int dim, int size)
{
	this->mesh = mesh;
	this->weights = weights;
	this->dim = dim;
	this->size = size;
}

StencilComponentN::~StencilComponentN() {}

void StencilComponentN::evaluate(float **output, int *outputSize)
{
	float *input;
	int inputSize;

	readIn(&input, &inputSize);

	//TODO
	*output = new float[5];
	for (int i=0; i<5; i++) (*output)[i] = i + 1;
	*outputSize = 5;
}

void StencilComponentN::readIn(float **arr, int *size)
{
	//TODO
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

	//TODO
	*output = new float[*size];
	for (int i=0; i<*size; i++) (*output)[i] = leftOut[i] + rightOut[i];
}



StencilN *Stencil(string mesh, Node *body)
{
	return new StencilN(mesh, body);
}

StencilComponentN *StencilComponent(string mesh, float *weights, int dim, int size)
{
	return new StencilComponentN(mesh, weights, dim, size);
}

StencilOpN *StencilOp(int op, Node *left, Node *right)
{
	return new StencilOpN(op, left, right);
}










