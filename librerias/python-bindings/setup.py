from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("pylibfromc",
              sources=["cython_binding.pyx", "cmult.c"],
              language="c",)

setup(name = "cython_pylibfromc",
      ext_modules = cythonize(ext))