import sys
sys.path.insert(0, '../../')

from snowflake.nodes import Stencil, WeightArray, StencilComponent, RectangularDomain
from snowflake.stencil_compiler import PythonCompiler
from snowflake.cuda_compiler import CUDACompiler

import numpy as np
from snowflake.vector import Vector


def print1DArray(array, size):
	for x in xrange(size):
		sys.stdout.write("{:5.2f}".format(array[x])); sys.stdout.write(" ")
	sys.stdout.write("\n")


#create input, output, and weight arrays
dataSize = 10
wSize = 3
weights = np.ones(wSize, dtype=np.float) / 3
print("Weight Array: "); print1DArray(weights, wSize); print("\n")

input = np.zeros(dataSize, dtype=np.float)
out = np.zeros_like(input)
for x in xrange(10):
	input[x] = x
print("Input: "); print1DArray(input, dataSize); print("\n")

#create stencil (Snowflake AST)
weight_component = StencilComponent(
	"input",
	WeightArray(weights)
)

stencil = Stencil(
	weight_component,
	"output",
	[(1, -1, 1)]
)

#compile
compiler = CUDACompiler()
kern = compiler.compile(stencil)

#execute
kern(out, input)

print("Output: "); print1DArray(out, dataSize); print("\n")




