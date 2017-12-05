from snowflake.nodes import *
from snowflake.compiler_nodes import *
from snowflake.stencil_compiler import Compiler

import sys
import numpy as np
from snowflake.vector import Vector
import ast
import subprocess
from array import *
from collections import Iterable



def getSizes(arr, lst):
			if not isinstance(arr, Iterable):
				return
			getSizes(arr[0], lst)
			lst.append(len(arr))


class CUDAApplicationGenerator(ast.NodeVisitor):

	def visit_StencilGroup(self, node):
		str = ""
		for s in node.body:
			str += self.visit(s) + "\n"
		return str

	def visit_Stencil(self, node):
		str = "Stencil(\"" + node.output + "\", " + self.visit(node.op_tree) + ");"
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
		dimAndsize = self.visit(node.weights)
		str = "StencilComponent(\"" + node.name + "\", " + node.name + "__WEIGHTS__, " + dimAndsize[0] + ", " + dimAndsize[1] + ", " + node.name + "__WEIGHTS_SIZES__)"
		return str

	def visit_WeightArray(self, node):
		sizeStr = str(len(node.weights))
		sizes = []
		getSizes(node.rawData, sizes)
		dimStr = str(len(sizes))
		return [dimStr, sizeStr]




class CUDAIOGenerator(ast.NodeVisitor):

	inputList = []
	outputList = []

	def __init__(self):
		self.inputList = []
		self.outputList = []

	def visit_StencilGroup(self, node):
		str = ""
		for s in node.body:
			str += self.visit(s) + "\n"
		return str

	def visit_Stencil(self, node):
		self.outputList.append(node.output)
		return self.visit(node.op_tree)

	def visit_StencilOp(self, node):
		str = self.visit(node.left)
		str += self.visit(node.right)
		return str

	def visit_StencilComponent(self, node):
		self.inputList.append(node.name)
		wstr = "float " + node.name + "__WEIGHTS__[] = " + self.visit(node.weights) + ";\n"
		
		sizes = []
		getSizes(node.weights.rawData, sizes)
		sStr = "int " + node.name + "__WEIGHTS_SIZES__[] = {" + str(sizes[0])
		for s in sizes[1:]:
			sStr += ", " + str(s)
		sStr += "};\n"
		
		return wstr + sStr

	def visit_WeightArray(self, node):
		wStr = "{" + str(node.weights[0].value)
		for sc in node.weights[1:]:
			wStr += ", " + str(float(sc.value))
		return wStr + "}"




class CUDACompiler(Compiler):
		
	def _post_process(self, original, compiled, index_name, **kwargs):

		'''print("Nodes: ")
		for node in ast.walk(original):
			if isinstance(node, StencilNode):
				print(type(node).__name__)
				pass
		print("")
        '''

		iog = CUDAIOGenerator()
		ioStr = iog.visit(original)
		appStr = CUDAApplicationGenerator().visit(original)

		#print("CUDA IO Code: \n") ; print(ioStr) ; print("")
		#print("CUDA Application Code: \n") ; print(appStr) ; print("")

		cudaSrcDir = "../../cuda"
		cudaPartsDir = "../../cuda-parts"
		workDir = "./cudaSrc"

		subprocess.call("mkdir -p " + workDir, shell=True)
		subprocess.call("rm -rf " + workDir + "/*", shell=True)
		" + workDir + "
		subprocess.call("cp -r " + cudaSrcDir + "/* ./" + workDir + "/", shell=True)
		
		subprocess.call("touch " + workDir + "/main.cu", shell=True)
		subprocess.call("cat " + cudaPartsDir + "/includes.cu.part >> " + workDir + "/main.cu", shell=True)
		subprocess.call("cat " + cudaPartsDir + "/extra-defs.cu.part >> " + workDir + "/main.cu", shell=True)
		subprocess.call("cat " + cudaPartsDir + "/main-start.cu.part >> " + workDir + "/main.cu", shell=True)
		f = open(workDir + "/main.cu", 'a') ; f.write(ioStr) ; f.close()
		f = open(workDir + "/main.cu", 'a') ; f.write(appStr) ; f.close()
		subprocess.call("cat " + cudaPartsDir + "/main-end.cu.part >> " + workDir + "/main.cu", shell=True)

		#execute cuda code
		#TODO
		

		def toCall(*args):

			inLen = len(iog.inputList)
			outLen = len(iog.outputList)
			argsNum= inLen + outLen
			if argsNum != len(args):
				print("Error: Number of arguments to the callable must be " + str(argsNum))
				return
			
			#print("Input List: ", iog.inputList)
			#print("Output List: ", iog.outputList)


			def flatten_helper(out, inp):

				if len(inp) == 0:
					return

				if not isinstance(inp[0], Iterable):
					out.append(inp[0])
				else:
					flatten_helper(out, inp[0])

				flatten_helper(out, inp[1:])

			def flatten(inp):
				out = []
				flatten_helper(out, inp)
				return out
				

			for i, fname in enumerate(iog.inputList):
				mdData = args[i + outLen]
				linData = flatten(args[i + outLen])
				dims = []
				getSizes(mdData, dims)

				sizeBin = array('f', [len(linData)])
				dimsBin = array('f', dims)
				dataBin = array('f', linData)

				f = open(workDir + "/" + fname, 'wb')
				sizeBin.tofile(f)
				dimsBin.tofile(f)
				dataBin.tofile(f)
				f.close()

			#execute cuda code
			#TODO

			for i, fname in enumerate(iog.outputList):
				f = open(fname, 'r')
				rData = array('f')
				rData.fromstring(f.read())
				f.close()

				data = rData.tolist()

				output = args[i]
				sizes = []
				getSizes(output, sizes)
				sizes.reverse()
				
				print(sizes)
				shaped = np.reshape(data, sizes)

				for i, item in enumerate(shaped):
					output[i] = item




		return toCall




