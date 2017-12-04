#include <iostream>
#include "nodes.h"

int main() {

float component1__WEIGHTS__[] = {1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
float component2__WEIGHTS__[] = {0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111};

Stencil("output", StencilOp(MUL, StencilComponent("component1", component1__WEIGHTS__, 3, 9), StencilComponent("component2", component2__WEIGHTS__, 3, 9)));

return 0;
}
