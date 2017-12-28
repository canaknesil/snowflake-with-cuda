import sys
sys.path.insert(0, '../../')

from snowflake.nodes import *
from snowflake.stencil_compiler import *
from snowflake.cuda_compiler import CUDACompiler

import numpy as np
from snowflake.vector import Vector

from PIL import Image
import time


pythonTime = time.time()
CUDATime = pythonTime
CTime = pythonTime


# apply stencil
weights = np.ones((9, 9), dtype=np.float) / 81
weight_componentR = StencilComponent("inputR", WeightArray(weights))
stencilR = Stencil(weight_componentR, "outputR", [(4, -4, 1)]*2)

CUDAkern = CUDACompiler().compile(stencilR)
ckern = CCompiler().compile(stencilR)
pythonkern = PythonCompiler().compile(stencilR)

def test(side):

	global pythonTime
	global CUDATime
	global CTime

	global pythonkern
	global ckern
	global CUDAkern
	

	pixR = np.ones((side, side), dtype=np.float)
	pixRout = np.zeros_like(pixR)

	start = time.time()
	pythonkern(pixRout, pixR)
	end = time.time()
	pythonTime = end - start

	start = time.time()
	ckern(pixRout, pixR)
	end = time.time()
	CTime = end - start

	start = time.time()
	CUDAkern(pixRout, pixR)
	end = time.time()
	CUDATime = end - start


for i in xrange(150, 4001, 150):
	test(i)
	print(str(i) + " " + str(pythonTime) + " " + str(CTime) + " " + str(CUDATime))

