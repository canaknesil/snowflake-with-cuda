#include <iostream>
#include "nodes.h"

int main() {

float inputR__WEIGHTS__[] = {0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123, 0.0123456790123};
int inputR__WEIGHTS_SIZES__[] = {9, 9};

Stencil("outputR", StencilComponent("inputR", inputR__WEIGHTS__, 2, 81, inputR__WEIGHTS_SIZES__));
return 0;
}
