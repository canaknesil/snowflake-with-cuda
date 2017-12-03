import sys
sys.path.append('../../')

from snowflake.nodes import *
from snowflake.compiler_nodes import *
from snowflake.stencil_compiler import Compiler

import numpy as np
from snowflake.vector import Vector
import ast


class StencilVisitor(ast.NodeVisitor):

	def visit_StencilGroup(self, node):
		print("StencilGroup stencil_list: ")
		for s in node.body:
			self.visit(s)

	def visit_Stencil(self, node):
		print("Stencil")
		self.visit(node.op_tree)

	def visit_StencilOp(self, node):
		print("StencilOp op: ", node.op, "left: ")
		self.visit(node.left)
		print("right: ")
		self.visit(node.right)

	def visit_StencilComponent(self, node):
		print("StencilComponent name: ", node.name, "Weights: ")
		self.visit(node.weights)

	def visit_WeightArray(self, node):
		print("Weight Array")


class MyCompiler(Compiler):
		
	def _post_process(self, original, compiled, index_name, **kwargs):

		print("Nodes: ")
		for node in ast.walk(original):
			if isinstance(node, StencilNode):
				print(type(node).__name__)
				pass
		print("")

		v = StencilVisitor()
		v.visit(original)

		def toCall():
			print("Callable is called.")

		return toCall


def evaluateSimpleStencil():
	#create input, output, and weight arrays
	wSizes = (3, 3)
	weights = np.ones(wSizes, dtype=np.float) / 9
	print("Weight Array: "); print(weights); print("\n")

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
	visitor = MyCompiler()
	kern = visitor.compile(stencil)
	kern()

def evaluateComplexStencil():
	#create input, output, and weight arrays
	dataSizes = (10, 10)
	wSizes = (3, 3)
	weights1 = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
	weights2 = np.ones(wSizes, dtype=np.float) / 9
	print("Weight Array 1: "); print(weights1); print("\n")
	print("Weight Array 2: "); print(weights2); print("\n")

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
	compiler = MyCompiler()
	kern = compiler.compile(stencil)
	kern()


evaluateSimpleStencil()
evaluateComplexStencil()



