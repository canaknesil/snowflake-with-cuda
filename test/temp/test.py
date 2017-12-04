import sys
sys.path.append('../../')

from snowflake.nodes import *
from snowflake.compiler_nodes import *
from snowflake.cuda_compiler import CUDACompiler

import numpy as np
from snowflake.vector import Vector


def evaluateSimpleStencil():
	#create input, output, and weight arrays
	wSizes = (3, 3)
	weights = np.ones(wSizes, dtype=np.float) / 9
	#print("Weight Array: "); print(weights); print("\n")

	#create stencil (Snowflake AST)
	weight_component = StencilComponent(
		"component",
		WeightArray(weights)
	)

	stencil = Stencil(
		weight_component,
		"output",
		[(1, -1, 1)]*2
	)

	#compile
	c = CUDACompiler()
	kern = c.compile(stencil)
	kern()

def evaluateComplexStencil():
	#create input, output, and weight arrays
	dataSizes = (10, 10)
	wSizes = (3, 3)
	weights1 = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
	weights2 = np.ones(wSizes, dtype=np.float) / 9
	#print("Weight Array 1: "); print(weights1); print("\n")
	#print("Weight Array 2: "); print(weights2); print("\n")

	#create stencil (Snowflake AST)
	weight_component1 = StencilComponent(
		"component1",
		WeightArray(weights1)
	)
	weight_component2 = StencilComponent(
		"component2",
		WeightArray(weights2)
	)

	stencil = Stencil(
		weight_component1 * weight_component2,
		"output",
		[(1, -1, 1)]*2
	)

	#compile
	c = CUDACompiler()
	kern = c.compile(stencil)
	kern()


def evaluateComplexStencil2():
	#create input, output, and weight arrays
	dataSizes = (10, 10)
	wSizes = (3, 3)
	weights1 = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
	weights2 = np.ones(wSizes, dtype=np.float) / 9
	#print("Weight Array 1: "); print(weights1); print("\n")
	#print("Weight Array 2: "); print(weights2); print("\n")

	#create stencil (Snowflake AST)
	weight_component1 = StencilComponent(
		"component1",
		WeightArray(weights1)
	)
	weight_component2 = StencilComponent(
		"component2",
		WeightArray(weights2)
	)

	stencil1 = Stencil(
		weight_component1,
		"output1",
		[(1, -1, 1)]*2
	)
	stencil2 = Stencil(
		weight_component2,
		"output2",
		[(1, -1, 1)]*2
	)

	stencilGroup = StencilGroup([stencil1, stencil2])

	#compile
	c = CUDACompiler()
	kern = c.compile(stencilGroup)
	kern()


#evaluateSimpleStencil()
evaluateComplexStencil()
#evaluateComplexStencil2()