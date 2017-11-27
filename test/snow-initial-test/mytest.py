import sys
sys.path.append('../../')

from snowflake.nodes import Stencil, WeightArray, StencilComponent, RectangularDomain
from snowflake.stencil_compiler import PythonCompiler

import numpy as np
from snowflake.vector import Vector

#create input, output, and weight arrays
weights = np.ones((3, 3), dtype=np.float)
print("Weight Array: "); print(weights); print("\n")

input = np.ones((7, 7), dtype=np.float)
out = np.zeros_like(input)
print("Input: "); print(input); print("\n")

#create stencil (Snowflake AST)
weight_component = StencilComponent(
	"weight_array",
	WeightArray(weights)
)

stencil = Stencil(
	weight_component,
	"output",
	[(1, -1, 1)]*2
)

#compile
compiler = PythonCompiler()
kern = compiler.compile(stencil)

#execute
kern(out, input)

print("Output: ")
print(out)




