#include <iostream>
#include "nodes.h"

int main() {

float input__WEIGHTS__[] = {0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111};
int input__WEIGHTS_SIZES__[] = {3, 3};

Stencil("output", StencilComponent("input", input__WEIGHTS__, 2, 9, input__WEIGHTS_SIZES__));

return 0;
}
