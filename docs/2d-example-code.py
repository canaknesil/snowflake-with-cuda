
wArr = WeightArray(ones((3, 3), dtype=float) / 9)
sC = StencilComponent("weight_array", wArr)
rD = RectangularDomain([(1, -1, 1)]*2)
s = Stencil(sC, "output", rD)

c = PythonCompiler()
# c = CCompiler()
# c = CUDACompiler()
callable = c.compile(s)

callable(output, input)