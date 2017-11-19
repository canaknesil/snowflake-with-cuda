
import sys
sys.path.append('../../snowflake')

from snowflake.nodes import Stencil, WeightArray, StencilComponent, SparseWeightArray
from snowflake.stencil_compiler import PythonCompiler

import numpy as np
from snowflake.vector import Vector



def run_test():
    weight_array = WeightArray(
        np.ones((3, 3, 3), dtype=np.float) / 27
    )
    component = StencilComponent(
        "input",
        weight_array
    )
    stencil = Stencil(
        component,
        "output",
        [(1, -1, 1)]*3
    )
    compiler = PythonCompiler()
    kern = compiler.compile(stencil)
    arr = np.arange(6**3, dtype=np.float).reshape((6, 6, 6))
    out = np.zeros_like(arr)
    kern(out, arr)

    print(out)


if __name__ == "__main__":
    run_test()
    