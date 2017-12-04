from snowflake.nodes import *
from snowflake.compiler_nodes import *
from snowflake.stencil_compiler import Compiler

import sys
import numpy as np
from snowflake.vector import Vector
import ast
import subprocess


class CUDAApplicationGenerator(ast.NodeVisitor):

	def visit_StencilGroup(self, node):
		str = ""
		for s in node.body:
			str += self.visit(s) + "\n"
		return str

	def visit_Stencil(self, node):
		str = "Stencil(" + node.output + ", " + self.visit(node.op_tree) + ");"
		return str

	def visit_StencilOp(self, node):
		def opSelect(o):
			return {
				operator.add: "ADD",
				operator.sub: "SUB",
				operator.mul: "MUL",
				operator.div: "DIV"
			}[o]
		opStr = opSelect(node.op)
		
		str = "StencilOp(" + opStr + ", " + self.visit(node.left) + ", " + self.visit(node.right) + ")"
		return str

	def visit_StencilComponent(self, node):
		str = "StencilComponent(" + node.name + ", " + node.name + "__WEIGHTS__)"
		return str

	def visit_WeightArray(self, node):
		pass


class CUDAIOGenerator(ast.NodeVisitor):

	def visit_StencilGroup(self, node):
		str = ""
		for s in node.body:
			str += self.visit(s) + "\n"
		return str

	def visit_Stencil(self, node):
		str = "float *" + node.output + " = new float[" + node.output + "__SIZE__];\n"
		str += self.visit(node.op_tree)
		return str

	def visit_StencilOp(self, node):
		str = self.visit(node.left)
		str += self.visit(node.right)
		return str

	def visit_StencilComponent(self, node):
		str = "float *" + node.name + " = new float[" + node.name + "__SIZE__];\n"
		str += "float *" + node.name + "__WEIGHTS__ = " + self.visit(node.weights) + ";\n"
		return str

	def visit_WeightArray(self, node):
		wStr = "{" + str(node.weights[0].value)
		for sc in node.weights[1:]:
			wStr += ", " + str(sc.value)
		return wStr + "}"



class CUDAIOFreeGenerator(ast.NodeVisitor):
	def visit_StencilGroup(self, node):
		str = ""
		for s in node.body:
			str += self.visit(s) + "\n"
		return str

	def visit_Stencil(self, node):
		str = "delete[] " + node.output + ";\n"
		str += self.visit(node.op_tree)
		return str

	def visit_StencilOp(self, node):
		str = self.visit(node.left)
		str += self.visit(node.right)
		return str

	def visit_StencilComponent(self, node):
		str = "delete[] " + node.name + ";\n"
		return str

	def visit_WeightArray(self, node):
		pass


class CUDASizeDefGenerator(ast.NodeVisitor):
	def visit_StencilGroup(self, node):
		str = ""
		for s in node.body:
			str += self.visit(s) + "\n"
		return str

	def visit_Stencil(self, node):
		return self.visit(node.op_tree)

	def visit_StencilOp(self, node):
		str = self.visit(node.left)
		str += self.visit(node.right)
		return str

	def visit_StencilComponent(self, node):
		dimAndsize = self.visit(node.weights)
		str = "#define " + node.name + "__WEIGHTS_SIZE__ " + dimAndsize[1] + "\n"
		str += "#define " + node.name + "__DIM__ " + dimAndsize[0] + "\n"
		return str

	def visit_WeightArray(self, node):
		sizeStr = str(len(node.weights))
		dimStr = str(len(node.rawData))
		return [dimStr, sizeStr]


class CUDACompiler(Compiler):
		
	def _post_process(self, original, compiled, index_name, **kwargs):

		'''print("Nodes: ")
		for node in ast.walk(original):
			if isinstance(node, StencilNode):
				print(type(node).__name__)
				pass
		print("")
        '''

		sizeDefStr = CUDASizeDefGenerator().visit(original)
		ioStr = CUDAIOGenerator().visit(original)
		appStr = CUDAApplicationGenerator().visit(original)
		ioFreeStr = CUDAIOFreeGenerator().visit(original)

		print("CUDA Size Definitions Code: \n") ; print(sizeDefStr) ; print("")
		print("CUDA IO Code: \n") ; print(ioStr) ; print("")
		print("CUDA Application Code: \n") ; print(appStr) ; print("")
		print("CUDA IO Free Code: \n") ; print(ioFreeStr) ; print("")

		cudaSrcDir = "../../cuda"
		cudaPartsDir = "../../cuda-parts"

		subprocess.call("mkdir -p cudaSrc", shell=True)
		subprocess.call("rm -rf cudaSrc/*", shell=True)
		
		subprocess.call("cp -r " + cudaSrcDir + "/* ./cudaSrc/", shell=True)
		
		subprocess.call("touch cudaSrc/main.cu", shell=True)
		subprocess.call("cat " + cudaPartsDir + "/includes.cu.part >> cudaSrc/main.cu", shell=True)
		f = open('cudaSrc/main.cu', 'a') ; f.write(sizeDefStr) ; f.close()
		'''
		subprocess.call("cat " + cudaPartsDir + "/extra-defs.cu.part >> cudaSrc/main.cu")
		subprocess.call("cat " + cudaPartsDir + "/main-start.cu.part >> cudaSrc/main.cu")
		f = open('cudaSrc/main.cu', 'a') ; f.write(ioStr) ; f.close()
		f = open('cudaSrc/main.cu', 'a') ; f.write(appStr) ; f.close()
		f = open('cudaSrc/main.cu', 'a') ; f.write(ioFreeStr) ; f.close()
		subprocess.call("cat " + cudaPartsDir + "/main-end.cu.part >> cudaSrc/main.cu")
		'''

		def toCall():
			print("Callable is called.")

		return toCall




