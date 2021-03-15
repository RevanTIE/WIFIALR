from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("pylibfromcpp",
              sources=["cython_example.pyx", "cppmult.cpp"],
              language="c++",)

setup(name = "cython_pylibfromcpp",
      ext_modules = cythonize(ext))