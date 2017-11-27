import sys
sys.path.append('../../')

from snowflake.nodes import Stencil, WeightArray, StencilComponent, RectangularDomain
from snowflake.stencil_compiler import PythonCompiler

import numpy as np
from snowflake.vector import Vector


def print2DArray(array, sizes):
	for x in xrange(sizes[0]):
		for y in xrange(sizes[1]):
			sys.stdout.write("{:5.2f}".format(array[x][y])); sys.stdout.write(" ")
		sys.stdout.write("\n")


#create input, output, and weight arrays
dataSizes = (10, 10)
wSizes = (3, 3)
weights = np.ones(wSizes, dtype=np.float) / 9
print("Weight Array: "); print2DArray(weights, wSizes); print("\n")

input = np.zeros(dataSizes, dtype=np.float)
out = np.zeros_like(input)
for x in xrange(10):
	for y in xrange(10):
		if x<5 and y<5: input[x][y] = 1
		if x<5 and y>=5: input[x][y] = 3
		if x>=5 and y<5: input[x][y] = 2
		if x>=5 and y>=5: input[x][y] = 4

print("Input: "); print2DArray(input, dataSizes); print("\n")

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

print("Output: "); print2DArray(out, dataSizes); print("\n")




