from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy, os, sys

if sys.platform == 'darwin':
    os.environ['CFLAGS']   = '-DGGML_USE_ACCELERATE -O3 -std=gnu11'
    os.environ['CXXFLAGS'] = '-DGGML_USE_ACCELERATE -O3 -std=c++11'
    os.environ['LDFLAGS']  = '-framework Accelerate'
else:
    os.environ['CFLAGS']   = '-mavx -mavx2 -mfma -mf16c -O3 /std:c++14'
    os.environ['CXXFLAGS'] = '-mavx -mavx2 -mfma -mf16c -O3 /std:c++14'

ext_modules = [
    Extension(
        name="whispercpp",
        sources=["C:\\Users\\aidan\\Desktop\\Code\\projectMesa\\whispercpp.py\\whispercpp.pyx", "C:\\Users\\aidan\\Desktop\\Code\\projectMesa\\whispercpp.py\\whisper.cpp/whisper.cpp"],
        language="c++",
        extra_compile_args=["/std:c++14"],
    )
]
ext_modules = cythonize(ext_modules)

whisper_clib = ('whisper_clib', {'sources': ['C:\\Users\\aidan\\Desktop\\Code\\projectMesa\\whispercpp.py\\whisper.cpp/ggml.c']})

setup(
    name='whispercpp',
    version='1.0',
    description='Python bindings for whisper.cpp',
    author='Luke Southam',
    author_email='luke@devthe.com',
    libraries=[whisper_clib],
    ext_modules=cythonize(ext_modules),
    include_dirs=['C:\\Users\\aidan\\Desktop\\Code\\projectMesa\\whispercpp.py\\whisper.cpp', numpy.get_include()],
    install_requires=[
      'numpy',
      'ffmpeg-python',
      'requests'
    ],
)
