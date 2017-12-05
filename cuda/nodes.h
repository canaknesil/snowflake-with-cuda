
#include <string>

using namespace std;


class Node
{
public:
	virtual void evaluate(float **output, int *size) = 0;
};


class StencilN : public Node
{
public:
	StencilN(string mesh, Node *body);
	~StencilN();

private:
	string mesh;
	Node *body;

	void evaluate(float **output, int *size);
	void writeOut(float *arr, int size);
};

StencilN *Stencil(string mesh, Node *body);


class StencilComponentN : public Node
{
public:
	StencilComponentN(string mesh, float *weights, int dim, int size, int *wSizes);
	~StencilComponentN();

	void evaluate(float **output, int *size);

private:
	string mesh;
	float *weights;
	int dim;
	int size;
	int *wSizes;

	void readIn(float **arr, int *size, int **dims);
};

StencilComponentN *StencilComponent(string mesh, float *weights, int dim, int size, int *wSizes);


#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3

class StencilOpN : public Node
{
public:
	StencilOpN(int op, Node *left, Node *right);
	~StencilOpN();

	void evaluate(float **output, int *size);

private:
	int op;
	Node *left, *right;
};

StencilOpN *StencilOp(int op, Node *left, Node *right);




