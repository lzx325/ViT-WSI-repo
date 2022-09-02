from distutils.core import setup, Extension
from Cython.Build import cythonize
ext=Extension(
    "slide_graph_builder.algos",
    sources=["slide_graph_builder/algos.pyx"]
)
setup(  
    ext_modules=cythonize([ext])
)