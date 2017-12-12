import sys
sys.path.insert(0, '../../')

from snowflake.nodes import *
from snowflake.stencil_compiler import PythonCompiler
from snowflake.cuda_compiler import CUDACompiler

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
	weights = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
	print("Weight Array: "); print2DArray(weights, wSizes); print("\n")

	input1 = np.zeros(dataSizes, dtype=np.float)
	outADD = np.zeros_like(input1)
	outSUB = np.zeros_like(input1)
	outMUL = np.zeros_like(input1)
	outDIV = np.zeros_like(input1)
	for x in xrange(10):
		for y in xrange(10):
			if x<5 and y<5: input1[x][y] = 1
			if x<5 and y>=5: input1[x][y] = 3
			if x>=5 and y<5: input1[x][y] = 2
			if x>=5 and y>=5: input1[x][y] = 4

	input2 = np.zeros(dataSizes, dtype=np.float)
	for x in xrange(10):
		for y in xrange(10):
			input2[x][y] = 2
			
	print("Input 1: "); print2DArray(input1, dataSizes); print("\n")
	print("Input 2: "); print2DArray(input2, dataSizes); print("\n")

	#create stencil (Snowflake AST)
	left_componentADD = StencilComponent(
		"left_componentADD",
		WeightArray(weights)
	)
	right_componentADD = StencilComponent(
		"right_componentADD",
		WeightArray(weights)
	)

	left_componentSUB = StencilComponent(
		"left_componentSUB",
		WeightArray(weights)
	)
	right_componentSUB = StencilComponent(
		"right_componentSUB",
		WeightArray(weights)
	)

	left_componentMUL = StencilComponent(
		"left_componentMUL",
		WeightArray(weights)
	)
	right_componentMUL = StencilComponent(
		"right_componentMUL",
		WeightArray(weights)
	)

	left_componentDIV = StencilComponent(
		"left_componentDIV",
		WeightArray(weights)
	)
	right_componentDIV = StencilComponent(
		"right_componentDIV",
		WeightArray(weights)
	)

	stencilADD = Stencil(
		left_componentADD + right_componentADD,
		"outputADD",
		[(1, -1, 1)]*2
	)
	stencilSUB = Stencil(
		left_componentSUB - right_componentSUB,
		"outputSUB",
		[(1, -1, 1)]*2
	)
	stencilMUL = Stencil(
		left_componentMUL * right_componentMUL,
		"outputMUL",
		[(1, -1, 1)]*2
	)
	stencilDIV = Stencil(
		left_componentDIV / right_componentDIV,
		"outputDIV",
		[(1, -1, 1)]*2
	)

	#compile
	compiler = CUDACompiler()
	kern = compiler.compile(StencilGroup([stencilADD, stencilSUB, stencilMUL, stencilDIV]))
	kern(outADD, outSUB, outMUL, outDIV, input1, input2, input1, input2, input1, input2, input1, input2)

	print("Output for ADD: "); print2DArray(outADD, dataSizes); print("\n")
	print("Output for SUB: "); print2DArray(outSUB, dataSizes); print("\n")
	print("Output for MUL: "); print2DArray(outMUL, dataSizes); print("\n")
	print("Output for DIV: "); print2DArray(outDIV, dataSizes); print("\n")



withStencilOp()


