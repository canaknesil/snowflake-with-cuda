
#include "nodes.h"

#define NULL 0


Stencil::Stencil(string mesh, Node body)
{
	this->mesh = mesh;
	this->body = body;
	evaluate(NULL, NULL);
}

Stencil::~Stencil() {}

void Stencil::evaluate(float *output, int *size)
{
	float *out;
	int size;

	body.evaluate(out, &size);

	writeOut(out, size);
}

void Stencil::writeOut(float *arr, int size)
{
	//TODO
}


StencilComponent::StencilComponent(string mesh, float* weights, int dim, int size)
{
	this->mesh = mesh;
	this->weights = weights;
	this->dim = dim;
	this->size = size;
}

StencilComponent::~StencilComponent() {}

void StencilComponent::evaluate(float *output, int *outputSize)
{
	float *input;
	int inputSize;

	readIn(input, &size);

	//TODO
	output = new float[5];
	for (int i=0; i<5; i++) output[i] = i + 1;
	*outputSize = 5;
}

void StencilComponent::readIn(float *arr, int *size)
{
	//TODO
}


StencilOp::StencilOp(int op, Node left, Node right)
{
	this->op = op;
	this->left = left;
	this->right = right;
}

StencilOp::~StencilOp() {}

void StencilOp::evaluate(float *output, int *size)
{
	float *leftOut, *rightOut;
	
	left.evaluate(leftOut, size);
	right.evaluate(rightOut, size);

	//TODO
	output = new float[*size];
	for (int i=0; i<*size; i++) output[i] = leftOut[i] + rightOut[i];
}

