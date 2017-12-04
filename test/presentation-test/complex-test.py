import sys
sys.path.append('../../')

from snowflake.nodes import *
from snowflake.stencil_compiler import PythonCompiler

import numpy as np
from snowflake.vector import Vector


def print2DArray(array, sizes):
	for x in xrange(sizes[0]):
		for y in xrange(sizes[1]):
			sys.stdout.write("{:5.2f}".format(array[x][y])); sys.stdout.write(" ")
		sys.stdout.write("\n")


def withStencilOp():
	#create input, output, and weight arrays
	dataSizes = (10, 10)
	wSizes = (3, 3)
	weights1 = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
	weights2 = np.ones(wSizes, dtype=np.float) / 9
	print("Weight Array 1: "); print2DArray(weights1, wSizes); print("\n")
	print("Weight Array 2: "); print2DArray(weights2, wSizes); print("\n")

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
	weight_component1 = StencilComponent(
		"weight_array1",
		WeightArray(weights1)
	)
	weight_component2 = StencilComponent(
		"weight_array2",
		WeightArray(weights2)
	)

	stencil = Stencil(
		weight_component1 + weight_component2,
		"output",
		[(1, -1, 1)]*2
	)

	#compile
	compiler = PythonCompiler()
	kern = compiler.compile(stencil)

	#execute
	kern(out, input, np.zeros(dataSizes, dtype=np.float))
	print("Output for weight array 1: "); print2DArray(out, dataSizes); print("\n")

	kern(out, np.zeros(dataSizes, dtype=np.float), input)
	print("Output for weight array 2: "); print2DArray(out, dataSizes); print("\n")

	kern(out, input, input)
	print("Output of sum of the results from both weight arrays: "); print2DArray(out, dataSizes); print("\n")



def withStencilGroup():
	#create input, output, and weight arrays
	dataSizes = (10, 10)
	wSizes = (3, 3)
	weights1 = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
	weights2 = np.ones(wSizes, dtype=np.float) / 9
	print("Weight Array 1: "); print2DArray(weights1, wSizes); print("\n")
	print("Weight Array 2: "); print2DArray(weights2, wSizes); print("\n")

	input = np.zeros(dataSizes, dtype=np.float)
	out1 = np.zeros_like(input)
	out2 = np.zeros_like(input)
	for x in xrange(10):
		for y in xrange(10):
			if x<5 and y<5: input[x][y] = 1
			if x<5 and y>=5: input[x][y] = 3
			if x>=5 and y<5: input[x][y] = 2
			if x>=5 and y>=5: input[x][y] = 4

	print("Input: "); print2DArray(input, dataSizes); print("\n")

	#create stencil (Snowflake AST)
	weight_component1 = StencilComponent(
		"weight_array1",
		WeightArray(weights1)
	)
	weight_component2 = StencilComponent(
		"weight_array2",
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
	compiler = PythonCompiler()
	kern = compiler.compile(stencilGroup)

	#execute
	kern(out1, out2, input, input)
	print("Output1: "); print2DArray(out1, dataSizes); print("\n")
	print("Output2: "); print2DArray(out2, dataSizes); print("\n")




withStencilOp()
withStencilGroup()