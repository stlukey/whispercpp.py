from distutils.core import setup
from Cython.Build import cythonize
import numpy, os, sys

if sys.platform == 'darwin':
    os.environ['CFLAGS']   = '-DGGML_USE_ACCELERATE -O3 -I./whisper.cpp/'
    os.environ['CXXFLAGS'] = '-DGGML_USE_ACCELERATE -O3 -I./whisper.cpp/'
    os.environ['LDFLAGS']  = '-framework Accelerate -I./whisper.cpp/'
else:
    os.environ['CFLAGS']   = '-mavx -mavx2 -mfma -mf16c -O3 -I./whisper.cpp/'
    os.environ['CXXFLAGS'] = '-mavx -mavx2 -mfma -mf16c -O3 -I./whisper.cpp/'


setup(
    name='whispercpp',
    version='1.0',
    description='Python bindings for whisper.cpp',
    author='Luke Southam',
    author_email='luke@devthe.com',
    ext_modules = cythonize("whispercpp.pyx"),
    include_dirs = ['./whisper.cpp/', numpy.get_include()],
    install_requires=[
      'numpy',
      'ffmpeg-python',
      'requests'
    ],
)
