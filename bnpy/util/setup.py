from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
	Extension("EntropyUtilX",
        ["EntropyUtilX.pyx"],
        libraries=["m"],
        extra_compile_args = ["-O3", "-ffast-math"])
	]

# CYTHON DIRECTIVES
# http://docs.cython.org/src/reference/compilation.html#compiler-directives
for e in ext_modules:
    e.cython_directives = {
		'embedsignature':True,
		'boundscheck':False,
		'nonecheck':False,
		'wraparound':False}

setup(
  name = "EntropyUtilX",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)

