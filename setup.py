from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os
import sys

if sys.platform == 'darwin':
    os.environ['CFLAGS'] = '-DGGML_USE_ACCELERATE'
    os.environ['CXXFLAGS'] = '-DGGML_USE_ACCELERATE'
    os.environ['LDFLAGS'] = '-framework Accelerate'
else:
    os.environ['CFLAGS'] = '-mavx -mavx2 -mfma -mf16c'
    os.environ['CXXFLAGS'] = '-mavx -mavx2 -mfma -mf16c'


setup(
    name='whispercpp',
    version='1.0',
    description='Python Bindings for whisper.cpp',
    author='Luke Southam',
    author_email='luke@devthe.com',
    ext_modules = cythonize("whispercpp.pyx"),
    include_dirs = ['./whisper.cpp/', numpy.get_include()],
    install_requires=[
      'numpy',
      'ffmpeg-python',
    ],
)
