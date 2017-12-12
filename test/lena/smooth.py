import sys
sys.path.insert(0, '../../')

from snowflake.nodes import *
from snowflake.stencil_compiler import PythonCompiler
from snowflake.cuda_compiler import CUDACompiler

import numpy as np
from snowflake.vector import Vector

from PIL import Image


# get input from image file
lena_noisy = Image.open("lena-noisy-small.jpg")
#lena_noisy = Image.open("lena-noisy.jpg")
sizes = lena_noisy.size
print(lena_noisy.bits, sizes, lena_noisy.format)

pix = np.array(lena_noisy)

pixR = np.ones(sizes, dtype=np.float)
pixG = np.ones(sizes, dtype=np.float)
pixB = np.ones(sizes, dtype=np.float)

for row in xrange(len(pix)):
	for col in xrange(len(pix[0])):
		pixR[row][col] = pix[row][col][0]
		pixG[row][col] = pix[row][col][1]
		pixB[row][col] = pix[row][col][2]

pixRout = np.zeros_like(pixR)
pixGout = np.zeros_like(pixR)
pixBout = np.zeros_like(pixR)


# apply stencil
weights = np.ones((5, 5), dtype=np.float) / 25

weight_componentR = StencilComponent("inputR", WeightArray(weights))
weight_componentG = StencilComponent("inputG", WeightArray(weights))
weight_componentB = StencilComponent("inputB", WeightArray(weights))

stencilR = Stencil(weight_componentR, "outputR", [(2, -2, 1)]*2)
stencilG = Stencil(weight_componentG, "outputG", [(2, -2, 1)]*2)
stencilB = Stencil(weight_componentB, "outputB", [(2, -2, 1)]*2)

sg = StencilGroup([stencilR, stencilG, stencilB])


compiler = CUDACompiler()
kern = compiler.compile(sg)

kern(pixRout, pixGout, pixBout, pixR, pixG, pixB)


# write output to image file
pixOut = np.zeros_like(pix)
for row in xrange(len(pixRout)):
	for col in xrange(len(pixRout[0])):
		pixOut[row][col] = [pixRout[row][col], pixGout[row][col], pixBout[row][col]]

pixOut = np.array(pixOut, np.uint8)


lena_smooth = Image.fromarray(pixOut)
lena_smooth.save("lena-smooth.jpg")