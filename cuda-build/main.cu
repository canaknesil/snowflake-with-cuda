#include <iostream>
#include "nodes.h"

int main() {

float left_componentADD__WEIGHTS__[] = {0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
int left_componentADD__WEIGHTS_SIZES__[] = {3, 3};
float right_componentADD__WEIGHTS__[] = {0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
int right_componentADD__WEIGHTS_SIZES__[] = {3, 3};

float left_componentSUB__WEIGHTS__[] = {0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
int left_componentSUB__WEIGHTS_SIZES__[] = {3, 3};
float right_componentSUB__WEIGHTS__[] = {0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
int right_componentSUB__WEIGHTS_SIZES__[] = {3, 3};

float left_componentMUL__WEIGHTS__[] = {0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
int left_componentMUL__WEIGHTS_SIZES__[] = {3, 3};
float right_componentMUL__WEIGHTS__[] = {0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
int right_componentMUL__WEIGHTS_SIZES__[] = {3, 3};

float left_componentDIV__WEIGHTS__[] = {0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
int left_componentDIV__WEIGHTS_SIZES__[] = {3, 3};
float right_componentDIV__WEIGHTS__[] = {0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
int right_componentDIV__WEIGHTS_SIZES__[] = {3, 3};

Stencil("outputADD", StencilOp(ADD, StencilComponent("left_componentADD", left_componentADD__WEIGHTS__, 2, 9, left_componentADD__WEIGHTS_SIZES__), StencilComponent("right_componentADD", right_componentADD__WEIGHTS__, 2, 9, right_componentADD__WEIGHTS_SIZES__)));
Stencil("outputSUB", StencilOp(SUB, StencilComponent("left_componentSUB", left_componentSUB__WEIGHTS__, 2, 9, left_componentSUB__WEIGHTS_SIZES__), StencilComponent("right_componentSUB", right_componentSUB__WEIGHTS__, 2, 9, right_componentSUB__WEIGHTS_SIZES__)));
Stencil("outputMUL", StencilOp(MUL, StencilComponent("left_componentMUL", left_componentMUL__WEIGHTS__, 2, 9, left_componentMUL__WEIGHTS_SIZES__), StencilComponent("right_componentMUL", right_componentMUL__WEIGHTS__, 2, 9, right_componentMUL__WEIGHTS_SIZES__)));
Stencil("outputDIV", StencilOp(DIV, StencilComponent("left_componentDIV", left_componentDIV__WEIGHTS__, 2, 9, left_componentDIV__WEIGHTS_SIZES__), StencilComponent("right_componentDIV", right_componentDIV__WEIGHTS__, 2, 9, right_componentDIV__WEIGHTS_SIZES__)));
return 0;
}

