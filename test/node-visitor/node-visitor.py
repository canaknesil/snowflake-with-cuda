import sys
sys.path.append('../../')

from snowflake.nodes import *
from snowflake.compiler_nodes import *
from snowflake.stencil_compiler import Compiler

import numpy as np
from snowflake.vector import Vector
import ast



class StencilVisitor(Compiler):

	def _evaluate(self, compiled):

		fields = [val for fname, val in ast.iter_fields(compiled)]
		
		if isinstance(compiled, IndexOp):
			print("IndexOp elts: ", fields[0], " ndim: ", fields[1], " name: ", fields[2])
		
		if isinstance(compiled, IterationSpace):
			print("IterationSpace space: ")
			self._evaluate(fields[0])
			for item in fields[1]:
				print("body: ")
				self._evaluate(item)
		
		if isinstance(compiled, NDSpace):
			print("NDSpace: ")
			for aspace in fields[0]:
				print("space: ")
				self._evaluate(aspace)
		
		if isinstance(compiled, Block):
			print("Block ")
			for item in fields[0]:
				print("body: ")
				self._evaluate(item)
		
		if isinstance(compiled, Space):
			print("Space low: ", fields[0], " high: ", fields[1], " stride: ", fields[2])
		
		if isinstance(compiled, NestedSpace):
			print("Space low: ", fields[0], " high: ", fields[1], " block_size: ", fields[2], " stride: ", fields[3])
		

	def _post_process(self, original, compiled, index_name, **kwargs):

		print("Nodes from orjinal: ")
		for node in ast.walk(original):
			if isinstance(node, ast.AST):
				print(type(node).__name__)
				pass

		print("\nNodes from compiled: ")
		for node in ast.walk(compiled):
			if isinstance(node, ast.AST):
				print(type(node).__name__)
				pass

		#print(ast.dump(original))
		#print(ast.dump(compiled))

		self._evaluate(compiled)

		def toCall():
			print("Callable is called.")

		return toCall


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
visitor = StencilVisitor()
kern = visitor.compile(stencil)
kern()



