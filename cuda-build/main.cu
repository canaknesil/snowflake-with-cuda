#include <iostream>
#include "nodes.h"

int main() {

float weight_array1__WEIGHTS__[] = {1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
int weight_array1__WEIGHTS_SIZES__[] = {3, 3};

float weight_array2__WEIGHTS__[] = {0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111, 0.111111111111};
int weight_array2__WEIGHTS_SIZES__[] = {3, 3};

Stencil("output1", StencilComponent("weight_array1", weight_array1__WEIGHTS__, 2, 9, weight_array1__WEIGHTS_SIZES__));
Stencil("output2", StencilComponent("weight_array2", weight_array2__WEIGHTS__, 2, 9, weight_array2__WEIGHTS_SIZES__));
return 0;
}

