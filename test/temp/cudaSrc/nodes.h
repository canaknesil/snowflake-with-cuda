
#include <string>


class Node
{
public:
	void evaluate(float *output, int *size) = 0;
};


class Stencil : public Node
{
public:
	Stencil(string mesh, Node body);
	~Stencil();

private:
	string mesh;
	Node body;

	void evaluate(float *output, int *size);
	void writeOut(float *arr, int size);
};


class StencilComponent : public Node
{
public:
	StencilComponent(string mesh, float *weights, int dim, int size);
	~StencilComponent();

	void evaluate(float *output, int *size);

private:
	string mesh;
	float *weights;
	int dim;
	int size;

	void readIn(float *arr, int *size);
};


#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3

class StencilOp : public Node
{
public:
	StencilOp(int op, Node left, Node right);
	~StencilOp();

	void evaluate(float *output, int *size);

private:
	int op;
	Node left, right;
};
