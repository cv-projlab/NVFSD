from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import os 


default_args=dict(
    language = "c", include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3"]
)

src = "nebfir/imop"

ext_modules = [
    Extension(
        name='nebfir.imop.or_scaling',
        sources=[f"{src}/or_scaling.pyx"],
        **default_args,),
    Extension(
        name='nebfir.imop.transformations',
        sources=[f"{src}/transformations.pyx"],
        **default_args)
]

setup(
    name="nebfir",
    cmdclass = {'build_ext': build_ext},
    ext_modules= ext_modules,
)
