import sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

args = ["-O2"] if sys.platform == "win32" else ["-O3", "-std=c++14", "-g", "-Wno-reorder"]

extension = Extension(
    "fast_cpu_nms",
    sources=["nms.pyx"],
    include_dirs=[numpy.get_include()],  # use numpy
    language="c++",
    extra_compile_args=args
)

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extension),
)