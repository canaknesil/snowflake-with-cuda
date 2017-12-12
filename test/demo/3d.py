import sys
sys.path.insert(0, '../../')

from snowflake.nodes import Stencil, WeightArray, StencilComponent, RectangularDomain
from snowflake.stencil_compiler import PythonCompiler
from snowflake.cuda_compiler import CUDACompiler

import numpy as np
from snowflake.vector import Vector


def print3DArray(array, sizes):
	for x in xrange(sizes[0]):
		for y in xrange(sizes[1]):
			for z in xrange(sizes[2]):
				sys.stdout.write("{:5.2f}".format(array[x][y][z])); sys.stdout.write(" ")
			sys.stdout.write("\n")
		sys.stdout.write("\n")
		sys.stdout.write("\n")


#create input, output, and weight arrays
dataSizes = (10, 10, 10)
wSizes = (3, 3, 3)
weights = np.ones(wSizes, dtype=np.float) / 27
print("Weight Array: "); print3DArray(weights, wSizes); print("\n")

input = np.zeros(dataSizes, dtype=np.float)
out = np.zeros_like(input)
for x in xrange(10):
	for y in xrange(10):
		for z in xrange(10):
			if x<5 and y<5: input[x][y][z] = 1
			if x<5 and y>=5: input[x][y][z] = 3
			if x>=5 and y<5: input[x][y][z] = 2
			if x>=5 and y>=5: input[x][y][z] = 4

print("Input: "); print3DArray(input, dataSizes); print("\n")

#create stencil (Snowflake AST)
weight_component = StencilComponent(
	"input",
	WeightArray(weights)
)

stencil = Stencil(
	weight_component,
	"output",
	[(1, -1, 1)]*3
)

#compile
compiler = CUDACompiler()
kern = compiler.compile(stencil)

#execute
kern(out, input)

print("Output: "); print3DArray(out, dataSizes); print("\n")




